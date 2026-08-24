from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.database import PROJECT_ROOT
from src.evaluate_model import (
    DEFAULT_AUDIT_JSON,
    DEFAULT_MODEL_ROOT,
    TARGET_PRECISION,
    TARGET_RECALL,
    best_by_f1,
    best_meeting_both,
    best_meeting_precision,
    best_meeting_recall,
    binary_metrics,
    load_missed_pairs,
    load_predictions,
    message_level_metrics,
    rule_baseline_prediction,
    search_thresholds,
)

DEFAULT_REPORT_PATH = PROJECT_ROOT / "reports" / "final_test_evaluation.md"


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def choose_val_threshold(val_rows) -> tuple[dict, str]:
    y_true = np.array([r.label for r in val_rows])
    scores = np.array([r.score for r in val_rows])
    grid = search_thresholds(y_true, scores)
    both = best_meeting_both(grid, TARGET_PRECISION, TARGET_RECALL)
    if both is not None:
        return both, "验证集上同时满足 Precision>=98% 与 Recall>=95%，取 F1 最高的阈值"
    recall_point = best_meeting_recall(grid, TARGET_RECALL)
    if recall_point is not None:
        return recall_point, "验证集上无法同时满足两个目标，取 Recall>=95% 里 Precision 最高的阈值"
    return best_by_f1(grid), "验证集上连 Recall>=95% 都达不到，取 F1 最高的阈值作为权宜之选"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="用验证集选定的阈值，在冻结测试集上做一次性最终评估（不在测试集上搜索阈值）"
    )
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--audit-json", type=Path, default=DEFAULT_AUDIT_JSON)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument(
        "--confirm-test",
        action="store_true",
        help="必须显式传入，确认这是对冻结测试集的一次性最终评估",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if not args.confirm_test:
        raise SystemExit("这是对冻结测试集的一次性评估，请显式加 --confirm-test 确认")

    missed_pairs = load_missed_pairs(args.audit_json)

    variant_dirs = sorted(
        p.parent
        for p in args.model_root.glob("*/predictions_test.jsonl")
        if (p.parent / "predictions_val.jsonl").is_file()
    )
    if not variant_dirs:
        raise SystemExit("找不到同时有 predictions_val.jsonl 和 predictions_test.jsonl 的变体")

    lines = ["# 冻结测试集最终评估（一次性，不做阈值搜索）", ""]
    lines.append(
        f"目标：Precision >= {TARGET_PRECISION:.0%}，Recall >= {TARGET_RECALL:.0%}（训练规划第9节）。"
        "阈值全部来自验证集搜索结果，测试集只用于套用阈值算最终指标，没有在测试集上做任何调参。"
    )
    lines.append("")
    lines.append("| 变体 | 验证集阈值 | 阈值选择依据 | 测试集 Precision | 测试集 Recall | 测试集 F1 | 规则基线(测试集) F1 |")
    lines.append("| --- | ---: | --- | ---: | ---: | ---: | ---: |")

    summary_rows = []
    for variant_dir in variant_dirs:
        name = variant_dir.name
        val_rows = load_predictions(variant_dir / "predictions_val.jsonl")
        test_rows = load_predictions(variant_dir / "predictions_test.jsonl")

        chosen_point, reason = choose_val_threshold(val_rows)
        threshold = chosen_point["threshold"]

        y_test = np.array([r.label for r in test_rows])
        scores_test = np.array([r.score for r in test_rows])
        y_pred_test = (scores_test >= threshold).astype(int)
        test_metrics = binary_metrics(y_test, y_pred_test)

        rule_pred_test = np.array(
            [rule_baseline_prediction(r, missed_pairs) for r in test_rows]
        )
        rule_metrics_test = binary_metrics(y_test, rule_pred_test)

        message_metrics = message_level_metrics(test_rows, threshold)

        summary_rows.append(
            {
                "variant": name,
                "threshold": threshold,
                "reason": reason,
                "val_metrics_at_threshold": {
                    "precision": chosen_point["precision"],
                    "recall": chosen_point["recall"],
                    "f1": chosen_point["f1"],
                },
                "test_metrics": test_metrics,
                "rule_baseline_test_metrics": rule_metrics_test,
                "message_level_metrics": message_metrics,
            }
        )

        lines.append(
            f"| {name} | {threshold:.4f} | {reason} | "
            f"{_fmt(test_metrics['precision'])} | {_fmt(test_metrics['recall'])} | "
            f"{_fmt(test_metrics['f1'])} | {_fmt(rule_metrics_test['f1'])} |"
        )

    lines.append("")
    lines.append("## 消息级别指标（测试集，各变体在各自阈值下）")
    lines.append("")
    lines.append(
        "| 变体 | 公司集合完全正确率 | 多公司消息完全正确率 | 漏报比例 | no_match误报率(抽样口径) |"
    )
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for row in summary_rows:
        m = row["message_level_metrics"]
        lines.append(
            f"| {row['variant']} | {_fmt(m['exact_match_rate'])} | "
            f"{_fmt(m['multi_company_exact_match_rate'])} | {_fmt(m['missed_any_rate'])} | "
            f"{_fmt(m['no_match_fp_rate_in_sample'])} |"
        )
    lines.append("")
    lines.append(
        "注：这里的“完全正确率”只覆盖候选生成阶段recall到的公司，候选召回率本身（约97.5%，"
        "见 `reports/data_audit.md`）会独立限制端到端整体正确率上限；no_match误报率只覆盖抽样进"
        "候选集合的背景负样本，不是全部无关消息，口径偏乐观。"
    )
    lines.append("")
    lines.append(
        "**测试集自本次评估起视为已使用。** 后续如需再看测试集表现，应该用新一轮标注数据重新"
        "切分测试集，而不是继续在这份测试集上调阈值或选模型。"
    )

    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text("\n".join(lines), encoding="utf-8", newline="\n")

    print(f"report -> {args.report_path}")
    print(json.dumps(summary_rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

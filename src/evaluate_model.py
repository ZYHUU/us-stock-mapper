from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.metrics import precision_recall_fscore_support

from src.database import PROJECT_ROOT

DEFAULT_MODEL_ROOT = PROJECT_ROOT / "models" / "company_classifier"
DEFAULT_AUDIT_JSON = PROJECT_ROOT / "reports" / "data_audit.json"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "reports" / "model_evaluation.md"
DEFAULT_THRESHOLD_JSON = PROJECT_ROOT / "models" / "company_classifier" / "threshold.json"

TARGET_PRECISION = 0.98
TARGET_RECALL = 0.95


@dataclass
class PredictionRow:
    message_id: int
    candidate_code: str
    label: int
    negative_type: str
    score: float


def load_predictions(path: Path) -> list[PredictionRow]:
    rows: list[PredictionRow] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            d = json.loads(line)
            rows.append(
                PredictionRow(
                    message_id=int(d["message_id"]),
                    candidate_code=str(d["candidate_code"]),
                    label=int(d["label"]),
                    negative_type=str(d["negative_type"]),
                    score=float(d["score"]),
                )
            )
    return rows


def load_missed_pairs(audit_json_path: Path) -> set[tuple[int, str]]:
    data = json.loads(audit_json_path.read_text(encoding="utf-8"))
    missed = data["candidate_recall"]["missed_pairs"]
    return {(int(item["message_id"]), str(item["code"])) for item in missed}


def rule_baseline_prediction(row: PredictionRow, missed_pairs: set[tuple[int, str]]) -> int:
    """如果只用规则/上游候选自己当分类器（"命中即算正确"），这条 pair 会被判成什么。"""
    if row.label == 1:
        return 0 if (row.message_id, row.candidate_code) in missed_pairs else 1
    return 1 if row.negative_type in ("hard_rule", "hard_upstream") else 0


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    return {"precision": float(precision), "recall": float(recall), "f1": float(f1)}


def search_thresholds(y_true: np.ndarray, scores: np.ndarray) -> list[dict]:
    thresholds = sorted(set(scores.tolist()) | {0.0, 1.0})
    grid = []
    for t in thresholds:
        y_pred = (scores >= t).astype(int)
        metrics = binary_metrics(y_true, y_pred)
        metrics["threshold"] = float(t)
        grid.append(metrics)
    return grid


def best_by_f1(grid: list[dict]) -> dict:
    return max(grid, key=lambda g: g["f1"])


def best_meeting_recall(grid: list[dict], min_recall: float) -> dict | None:
    candidates = [g for g in grid if g["recall"] >= min_recall]
    return max(candidates, key=lambda g: g["precision"]) if candidates else None


def best_meeting_precision(grid: list[dict], min_precision: float) -> dict | None:
    candidates = [g for g in grid if g["precision"] >= min_precision]
    return max(candidates, key=lambda g: g["recall"]) if candidates else None


def best_meeting_both(grid: list[dict], min_precision: float, min_recall: float) -> dict | None:
    candidates = [
        g for g in grid if g["precision"] >= min_precision and g["recall"] >= min_recall
    ]
    return max(candidates, key=lambda g: g["f1"]) if candidates else None


def message_level_metrics(rows: list[PredictionRow], threshold: float) -> dict:
    by_message: dict[int, list[PredictionRow]] = defaultdict(list)
    for row in rows:
        by_message[row.message_id].append(row)

    exact_match = 0
    exact_match_multi = 0
    multi_total = 0
    missed_any = 0
    positive_message_total = 0
    no_match_fp = 0
    no_match_total = 0

    for group in by_message.values():
        true_codes = {r.candidate_code for r in group if r.label == 1}
        predicted_codes = {r.candidate_code for r in group if r.score >= threshold}

        if true_codes:
            positive_message_total += 1
            if predicted_codes == true_codes:
                exact_match += 1
            if not true_codes.issubset(predicted_codes):
                missed_any += 1
            if len(true_codes) >= 2:
                multi_total += 1
                if predicted_codes == true_codes:
                    exact_match_multi += 1
        else:
            no_match_total += 1
            if predicted_codes:
                no_match_fp += 1

    def rate(n: int, d: int) -> float | None:
        return n / d if d else None

    return {
        "positive_message_total": positive_message_total,
        "exact_match_rate": rate(exact_match, positive_message_total),
        "multi_company_message_total": multi_total,
        "multi_company_exact_match_rate": rate(exact_match_multi, multi_total),
        "missed_any_rate": rate(missed_any, positive_message_total),
        "no_match_message_total_in_sample": no_match_total,
        "no_match_fp_rate_in_sample": rate(no_match_fp, no_match_total),
    }


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="在验证集上做阈值搜索、模型对比和消息级别评估")
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--audit-json", type=Path, default=DEFAULT_AUDIT_JSON)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--threshold-json", type=Path, default=DEFAULT_THRESHOLD_JSON)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    missed_pairs = load_missed_pairs(args.audit_json)

    variant_dirs = sorted(
        p.parent for p in args.model_root.glob("*/predictions_val.jsonl")
    )
    if not variant_dirs:
        raise SystemExit(f"在 {args.model_root} 下找不到任何 predictions_val.jsonl，请先运行 train_classic_model.py")

    variant_reports: dict[str, dict] = {}
    rule_metrics_by_variant: dict[str, dict] = {}

    for variant_dir in variant_dirs:
        variant_name = variant_dir.name
        rows = load_predictions(variant_dir / "predictions_val.jsonl")
        y_true = np.array([r.label for r in rows])
        scores = np.array([r.score for r in rows])

        grid = search_thresholds(y_true, scores)
        f1_point = best_by_f1(grid)
        recall_point = best_meeting_recall(grid, TARGET_RECALL)
        precision_point = best_meeting_precision(grid, TARGET_PRECISION)
        both_point = best_meeting_both(grid, TARGET_PRECISION, TARGET_RECALL)

        rule_pred = np.array([rule_baseline_prediction(r, missed_pairs) for r in rows])
        rule_metrics_by_variant[variant_name] = binary_metrics(y_true, rule_pred)

        variant_reports[variant_name] = {
            "rows": rows,
            "best_f1_point": f1_point,
            "recall95_point": recall_point,
            "precision98_point": precision_point,
            "both_targets_point": both_point,
        }

    def variant_rank_key(name: str) -> tuple[float, float]:
        report = variant_reports[name]
        both = report["both_targets_point"]
        if both is not None:
            return (2.0, both["f1"])
        recall_point = report["recall95_point"]
        if recall_point is not None:
            return (1.0, recall_point["precision"])
        return (0.0, report["best_f1_point"]["f1"])

    recommended_variant = max(variant_reports, key=variant_rank_key)
    recommended_report = variant_reports[recommended_variant]
    chosen_point = (
        recommended_report["both_targets_point"]
        or recommended_report["recall95_point"]
        or recommended_report["best_f1_point"]
    )
    chosen_reason = (
        "同时满足 Precision>=98% 与 Recall>=95%，取其中 F1 最高的阈值"
        if recommended_report["both_targets_point"] is not None
        else (
            "无法同时满足两个目标，退而求其次：在 Recall>=95% 的阈值里选 Precision 最高的"
            if recommended_report["recall95_point"] is not None
            else "验证集上没有阈值能达到 Recall>=95%，暂用 F1 最高的阈值作为权宜之选"
        )
    )

    message_metrics = message_level_metrics(
        recommended_report["rows"], chosen_point["threshold"]
    )

    args.threshold_json.parent.mkdir(parents=True, exist_ok=True)
    threshold_payload = {
        "recommended_variant": recommended_variant,
        "model_path": str(
            (args.model_root / recommended_variant / "model.joblib").resolve()
        ),
        "threshold": chosen_point["threshold"],
        "reason": chosen_reason,
        "validation_metrics_at_threshold": {
            "precision": chosen_point["precision"],
            "recall": chosen_point["recall"],
            "f1": chosen_point["f1"],
        },
        "message_level_metrics": message_metrics,
        "target_precision": TARGET_PRECISION,
        "target_recall": TARGET_RECALL,
    }
    args.threshold_json.write_text(
        json.dumps(threshold_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    lines = ["# 模型评估与阈值搜索（验证集）", ""]
    lines.append(
        f"目标：Precision >= {TARGET_PRECISION:.0%}，Recall >= {TARGET_RECALL:.0%}"
        "（来自训练规划第9节，第一版目标值，不是硬性保证）。"
    )
    lines.append("")
    lines.append("## 各变体对比（每个变体各自的最优可行阈值点）")
    lines.append("")
    lines.append(
        "| 变体 | 规则单独运行 P/R/F1 | 阈值搜索最优F1点 | 同时满足两目标的点 | 仅满足Recall>=95%时的最高Precision |"
    )
    lines.append("| --- | --- | --- | --- | --- |")
    for name, report in sorted(variant_reports.items()):
        rule = rule_metrics_by_variant[name]
        f1_point = report["best_f1_point"]
        both = report["both_targets_point"]
        recall_point = report["recall95_point"]
        lines.append(
            f"| {name} "
            f"| P={_fmt(rule['precision'])} R={_fmt(rule['recall'])} F1={_fmt(rule['f1'])} "
            f"| thr={f1_point['threshold']:.3f} P={_fmt(f1_point['precision'])} "
            f"R={_fmt(f1_point['recall'])} F1={_fmt(f1_point['f1'])} "
            f"| {'thr=' + format(both['threshold'], '.3f') + ' P=' + _fmt(both['precision']) + ' R=' + _fmt(both['recall']) if both else '无'} "
            f"| {'thr=' + format(recall_point['threshold'], '.3f') + ' P=' + _fmt(recall_point['precision']) if recall_point else '无'} |"
        )
    lines.append("")
    lines.append(
        "“规则单独运行”把候选生成阶段（本地规则+上游 stocks/stoks）本身当分类器：只要候选被召回就判正确，"
        "用来衡量分类器相对纯规则到底改进了多少。"
    )
    lines.append("")

    lines.append(f"## 推荐配置：`{recommended_variant}`")
    lines.append("")
    lines.append(f"选择依据：{chosen_reason}")
    lines.append("")
    lines.append(
        f"固定阈值：**{chosen_point['threshold']:.4f}**，"
        f"验证集 Precision={chosen_point['precision']:.3f}，"
        f"Recall={chosen_point['recall']:.3f}，F1={chosen_point['f1']:.3f}"
    )
    lines.append("")
    lines.append("## 消息级别指标（在推荐阈值下，仅覆盖本次候选集合内的 pair，不是完整121家公司）")
    lines.append("")
    lines.append("| 指标 | 数值 |")
    lines.append("| --- | ---: |")
    lines.append(f"| 含正确公司的消息数 | {message_metrics['positive_message_total']} |")
    lines.append(f"| 公司集合完全正确率 | {_fmt(message_metrics['exact_match_rate'])} |")
    lines.append(f"| 多公司消息数 | {message_metrics['multi_company_message_total']} |")
    lines.append(
        f"| 多公司消息完全正确率 | {_fmt(message_metrics['multi_company_exact_match_rate'])} |"
    )
    lines.append(f"| 漏掉任意一家公司的比例 | {_fmt(message_metrics['missed_any_rate'])} |")
    lines.append(
        f"| no_match 消息数（仅本次抽样的 background 负样本，非全部无关消息） | "
        f"{message_metrics['no_match_message_total_in_sample']} |"
    )
    lines.append(
        f"| no_match 误报率（同上，仅抽样口径） | "
        f"{_fmt(message_metrics['no_match_fp_rate_in_sample'])} |"
    )
    lines.append("")
    lines.append(
        "注：这里的 no_match 统计只覆盖 `build_training_dataset.py` 里抽样进候选集合的那部分背景消息"
        "（详见训练规划第6节的抽样说明），不是全部5292条无目标公司消息，口径偏乐观，"
        "不能直接当作生产环境的 no_match 误报率。"
    )
    lines.append("")
    lines.append(
        "注：这里的“完全正确率”只覆盖候选生成阶段recall到的公司；候选召回率本身（约97.5%，"
        "见 `reports/data_audit.md`）会独立限制端到端的整体正确率上限。"
    )

    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text("\n".join(lines), encoding="utf-8", newline="\n")

    print(f"report -> {args.report_path}")
    print(f"threshold -> {args.threshold_json}")
    print(json.dumps(threshold_payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

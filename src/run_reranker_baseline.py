from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.database import PROJECT_ROOT
from src.train_classic_model import evaluate_at_threshold, load_pairs

DEFAULT_PAIRS_PATH = PROJECT_ROOT / "data" / "training" / "candidate_pairs.jsonl"
DEFAULT_MODEL_ROOT = PROJECT_ROOT / "models" / "company_classifier"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "reports" / "reranker_zeroshot_report.md"
MODEL_NAME = "BAAI/bge-reranker-base"
VARIANT_NAME = "bge_reranker_zeroshot"
MAX_LENGTH = 256
BATCH_SIZE = 16
LATENCY_SAMPLE_SIZE = 50


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="跑 BAAI/bge-reranker-base 的零样本（未微调）基线，并测量CPU推理延迟"
    )
    parser.add_argument("--pairs-path", type=Path, default=DEFAULT_PAIRS_PATH)
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--split", default="val", choices=["val", "test"])
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--max-length", type=int, default=MAX_LENGTH)
    parser.add_argument(
        "--confirm-test",
        action="store_true",
        help="在测试集上运行时必须显式传入，避免误触碰冻结测试集",
    )
    return parser


def percentile(values: list[float], p: float) -> float:
    return float(np.percentile(np.array(values), p))


def main() -> None:
    args = _build_parser().parse_args()
    if args.split == "test" and not args.confirm_test:
        raise SystemExit("测试集当前应保持冻结，如需在测试集上运行请显式加 --confirm-test")

    examples = [e for e in load_pairs(args.pairs_path) if e.split == args.split]
    if not examples:
        raise SystemExit(f"split={args.split} 没有样本")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.eval()
    torch.set_num_threads(max(1, torch.get_num_threads()))

    scores: list[float] = []
    single_pair_latencies_ms: list[float] = []
    batch_latencies_ms: list[float] = []
    batch_per_item_latencies_ms: list[float] = []

    with torch.no_grad():
        # 预热：排除首次前向传播的懒加载/算子初始化开销，避免污染延迟统计
        for example in examples[:3]:
            inputs = tokenizer(
                [example.message_text], [example.candidate_profile],
                padding=True, truncation=True, max_length=args.max_length, return_tensors="pt",
            )
            model(**inputs)

        # 单候选延迟：模拟一条消息只召回1个候选公司的最简单场景
        for example in examples[:LATENCY_SAMPLE_SIZE]:
            start = time.perf_counter()
            inputs = tokenizer(
                [example.message_text], [example.candidate_profile],
                padding=True, truncation=True, max_length=args.max_length, return_tensors="pt",
            )
            model(**inputs)
            single_pair_latencies_ms.append((time.perf_counter() - start) * 1000)

        # 批量延迟：模拟一条消息有多个候选、批量打分的场景
        for start_idx in range(0, len(examples), args.batch_size):
            batch = examples[start_idx : start_idx + args.batch_size]
            messages = [e.message_text for e in batch]
            candidates = [e.candidate_profile for e in batch]
            start = time.perf_counter()
            inputs = tokenizer(
                messages, candidates, padding=True, truncation=True,
                max_length=args.max_length, return_tensors="pt",
            )
            logits = model(**inputs).logits.view(-1).float()
            elapsed_ms = (time.perf_counter() - start) * 1000
            batch_latencies_ms.append(elapsed_ms)
            batch_per_item_latencies_ms.append(elapsed_ms / len(batch))
            scores.extend(torch.sigmoid(logits).tolist())

    y_true = np.array([e.label for e in examples])
    scores_arr = np.array(scores)
    metrics_at_half = evaluate_at_threshold(y_true, scores_arr, threshold=0.5)

    variant_dir = args.model_root / VARIANT_NAME
    variant_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = variant_dir / f"predictions_{args.split}.jsonl"
    with predictions_path.open("w", encoding="utf-8", newline="\n") as file:
        for example, score in zip(examples, scores):
            file.write(
                json.dumps(
                    {
                        "message_id": example.message_id,
                        "candidate_code": example.candidate_code,
                        "label": example.label,
                        "negative_type": example.negative_type,
                        "score": float(score),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    latency_summary = {
        "single_pair_ms": {
            "p50": percentile(single_pair_latencies_ms, 50),
            "p95": percentile(single_pair_latencies_ms, 95),
            "p99": percentile(single_pair_latencies_ms, 99),
            "sample_size": len(single_pair_latencies_ms),
        },
        "batch_total_ms": {
            "p50": percentile(batch_latencies_ms, 50),
            "p95": percentile(batch_latencies_ms, 95),
            "p99": percentile(batch_latencies_ms, 99),
            "batch_size": args.batch_size,
            "num_batches": len(batch_latencies_ms),
        },
        "batch_per_item_ms": {
            "p50": percentile(batch_per_item_latencies_ms, 50),
            "p95": percentile(batch_per_item_latencies_ms, 95),
            "p99": percentile(batch_per_item_latencies_ms, 99),
        },
    }

    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# {MODEL_NAME} 零样本（未微调）基线 —— split={args.split}",
        "",
        f"样本数：{len(examples)}，阈值=0.5 粗略指标（正式阈值搜索见 evaluate_model.py）",
        "",
        "| Precision | Recall | F1 | TP | FP | FN | TN |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        f"| {metrics_at_half['precision']:.3f} | {metrics_at_half['recall']:.3f} | "
        f"{metrics_at_half['f1']:.3f} | {metrics_at_half['tp']} | {metrics_at_half['fp']} | "
        f"{metrics_at_half['fn']} | {metrics_at_half['tn']} |",
        "",
        "## CPU 推理延迟（本机测得，仅供参考，不是生产环境标定值）",
        "",
        "| 场景 | p50 | p95 | p99 |",
        "| --- | ---: | ---: | ---: |",
        f"| 单候选前向传播（每条消息仅1个候选，n={latency_summary['single_pair_ms']['sample_size']}） | "
        f"{latency_summary['single_pair_ms']['p50']:.1f}ms | "
        f"{latency_summary['single_pair_ms']['p95']:.1f}ms | "
        f"{latency_summary['single_pair_ms']['p99']:.1f}ms |",
        f"| 批量前向传播总耗时（batch_size={args.batch_size}） | "
        f"{latency_summary['batch_total_ms']['p50']:.1f}ms | "
        f"{latency_summary['batch_total_ms']['p95']:.1f}ms | "
        f"{latency_summary['batch_total_ms']['p99']:.1f}ms |",
        "| 批量场景下均摊到每个候选 | "
        f"{latency_summary['batch_per_item_ms']['p50']:.1f}ms | "
        f"{latency_summary['batch_per_item_ms']['p95']:.1f}ms | "
        f"{latency_summary['batch_per_item_ms']['p99']:.1f}ms |",
        "",
        "注：这里只测了模型前向传播本身，不包含候选召回、文本预处理和SQLite代码映射，"
        "不是训练规划第9节要求的完整端到端延迟；只用于判断这个模型量级本身是否可能进入100ms预算。",
    ]
    args.report_path.write_text("\n".join(lines), encoding="utf-8", newline="\n")

    print(f"predictions -> {predictions_path}")
    print(f"report -> {args.report_path}")
    print(json.dumps({"metrics_at_0.5": metrics_at_half, "latency_ms": latency_summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

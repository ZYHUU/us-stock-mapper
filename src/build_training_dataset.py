from __future__ import annotations

import argparse
import json
import random
import sqlite3
from collections import defaultdict
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.audit_training_data import (
    LabeledMessage,
    load_labeled_messages,
    load_upstream_codes_by_message,
    resolve_snapshot,
)
from src.database import PROJECT_ROOT
from src.mapper import Company, USStockMapper, load_enabled_companies

DEFAULT_SNAPSHOT_ROOT = PROJECT_ROOT / "data" / "snapshots"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "training"

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
# 剩余部分自动归入测试集

MIN_NEGATIVES_PER_POSITIVE_MESSAGE = 3
BACKGROUND_NEGATIVES_PER_MESSAGE = 1
RANDOM_SEED = 20260821


def _readonly_connection(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    return connection


def load_message_times(connection: sqlite3.Connection) -> dict[int, str]:
    rows = connection.execute(
        "SELECT id, published_at, received_at FROM messages"
    ).fetchall()
    times: dict[int, str] = {}
    for row in rows:
        published_at = str(row["published_at"] or "").strip()
        times[int(row["id"])] = published_at or str(row["received_at"] or "")
    return times


def build_candidate_profile(company: Company) -> str:
    alias_preview = ", ".join(company.aliases[:5])
    return f"{company.company_name} | {company.canonical_code} | aliases: {alias_preview}"


@dataclass
class PairRecord:
    message_id: int
    source_id: str
    source_type: str
    message_text: str
    candidate_code: str
    candidate_profile: str
    label: int
    negative_type: str
    annotation_confidence: str
    real_candidate_count: int
    split: str = ""


def build_pairs(
    messages: list[LabeledMessage],
    upstream_by_message: dict[int, list[str]],
    companies: list[Company],
    ticker_to_code: dict[str, str],
    code_to_company: dict[str, Company],
    rng: random.Random,
    background_sample_size: int,
) -> list[PairRecord]:
    mapper = USStockMapper(companies)
    all_codes = list(code_to_company.keys())

    candidate_labels_by_message: dict[int, dict[str, tuple[int, str]]] = {}
    # 候选数量特征必须反映“规则+上游真实召回了多少候选”，不能把后面填充的
    # random_topup/background 负样本也算进去——那些是为了凑够负样本数量人为
    # 加入的，训练和线上推理的候选数量算法必须一致，否则模型会学到训练数据
    # 构造过程留下的假象而不是真实信号（见本轮踩过的坑）。
    real_candidate_count_by_message: dict[int, int] = {}
    background_eligible: list[int] = []

    for m in messages:
        local_matches = {match.canonical_code for match in mapper.identify(m.text)}
        upstream_codes = {
            ticker_to_code[t]
            for t in upstream_by_message.get(m.message_id, [])
            if t in ticker_to_code
        }
        correct = set(m.correct_codes)

        candidate_labels: dict[str, tuple[int, str]] = {}
        for code in correct:
            candidate_labels[code] = (1, "")
        for code in local_matches - correct:
            candidate_labels[code] = (0, "hard_rule")
        for code in upstream_codes - correct - local_matches:
            candidate_labels[code] = (0, "hard_upstream")

        real_candidate_count_by_message[m.message_id] = len(candidate_labels)

        if correct:
            existing_negative_count = sum(
                1 for label, _ in candidate_labels.values() if label == 0
            )
            shortfall = MIN_NEGATIVES_PER_POSITIVE_MESSAGE - existing_negative_count
            if shortfall > 0:
                exclude = set(candidate_labels)
                pool = [code for code in all_codes if code not in exclude]
                rng.shuffle(pool)
                for code in pool[:shortfall]:
                    candidate_labels[code] = (0, "random_topup")
        elif not candidate_labels:
            background_eligible.append(m.message_id)

        candidate_labels_by_message[m.message_id] = candidate_labels

    # 没有目标公司、也没有任何候选命中的消息（"纯背景"消息）数量远超正样本，
    # 只抽样一小部分作为普通负样本，避免这一类别压倒其余类别（见训练规划第6节）。
    sampled_background = set(
        rng.sample(
            background_eligible, min(background_sample_size, len(background_eligible))
        )
    )
    for message_id in sampled_background:
        pool = list(all_codes)
        rng.shuffle(pool)
        candidate_labels_by_message[message_id][pool[0]] = (0, "background")

    records: list[PairRecord] = []
    for m in messages:
        for code, (label, negative_type) in candidate_labels_by_message[m.message_id].items():
            company = code_to_company[code]
            records.append(
                PairRecord(
                    message_id=m.message_id,
                    source_id=m.source_id,
                    source_type=m.source_type,
                    message_text=m.text,
                    candidate_code=code,
                    candidate_profile=build_candidate_profile(company),
                    label=label,
                    negative_type=negative_type,
                    annotation_confidence=m.confidence,
                    real_candidate_count=real_candidate_count_by_message[m.message_id],
                )
            )
    return records


def assign_splits(
    records: list[PairRecord], message_times: dict[int, str]
) -> dict[str, list[int]]:
    text_to_message_ids: dict[str, set[int]] = defaultdict(set)
    message_id_to_text: dict[int, str] = {}
    for r in records:
        text_to_message_ids[r.message_text].add(r.message_id)
        message_id_to_text[r.message_id] = r.message_text

    pair_count_by_message: dict[int, int] = defaultdict(int)
    for r in records:
        pair_count_by_message[r.message_id] += 1

    units: list[dict[str, Any]] = []
    for text, message_ids in text_to_message_ids.items():
        unit_time = max(message_times.get(mid, "") for mid in message_ids)
        unit_pairs = sum(pair_count_by_message[mid] for mid in message_ids)
        units.append({"message_ids": message_ids, "time": unit_time, "pairs": unit_pairs})

    units.sort(key=lambda u: u["time"])

    total_pairs = sum(u["pairs"] for u in units)
    train_target = total_pairs * TRAIN_RATIO
    val_target = total_pairs * (TRAIN_RATIO + VAL_RATIO)

    split_by_message: dict[int, str] = {}
    cumulative = 0
    for unit in units:
        cumulative += unit["pairs"]
        if cumulative <= train_target:
            split = "train"
        elif cumulative <= val_target:
            split = "val"
        else:
            split = "test"
        for mid in unit["message_ids"]:
            split_by_message[mid] = split

    grouped: dict[str, list[int]] = {"train": [], "val": [], "test": []}
    for mid, split in split_by_message.items():
        grouped[split].append(mid)
    for split in grouped:
        grouped[split].sort()
    return grouped


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="从标注数据生成消息-候选公司文本对训练数据集")
    parser.add_argument("--snapshot-root", type=Path, default=DEFAULT_SNAPSHOT_ROOT)
    parser.add_argument("--snapshot-name", required=True, help="要使用的固定快照名称")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument(
        "--background-sample-size",
        type=int,
        default=None,
        help=(
            "无候选、无目标公司的纯背景消息中抽样多少条作为普通负样本，"
            "默认等于正样本消息数量"
        ),
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    database_path, companies_path, manifest = resolve_snapshot(
        args.snapshot_root, args.snapshot_name
    )

    companies = load_enabled_companies(companies_path, database_path)
    ticker_to_code = {c.ticker: c.canonical_code for c in companies}
    code_to_company = {c.canonical_code: c for c in companies}

    with closing(_readonly_connection(database_path)) as connection:
        messages = load_labeled_messages(connection)
        upstream_by_message = load_upstream_codes_by_message(connection)
        message_times = load_message_times(connection)

    rng = random.Random(args.seed)
    positive_message_count = sum(1 for m in messages if m.correct_codes)
    background_sample_size = (
        args.background_sample_size
        if args.background_sample_size is not None
        else positive_message_count
    )
    records = build_pairs(
        messages,
        upstream_by_message,
        companies,
        ticker_to_code,
        code_to_company,
        rng,
        background_sample_size,
    )

    message_ids_by_split = assign_splits(records, message_times)
    message_to_split = {
        mid: split for split, ids in message_ids_by_split.items() for mid in ids
    }
    for r in records:
        r.split = message_to_split[r.message_id]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pairs_path = args.output_dir / "candidate_pairs.jsonl"
    with pairs_path.open("w", encoding="utf-8", newline="\n") as file:
        for r in records:
            file.write(
                json.dumps(
                    {
                        "message_id": r.message_id,
                        "source_id": r.source_id,
                        "source_type": r.source_type,
                        "message_text": r.message_text,
                        "candidate_code": r.candidate_code,
                        "candidate_profile": r.candidate_profile,
                        "label": r.label,
                        "negative_type": r.negative_type,
                        "annotation_confidence": r.annotation_confidence,
                        "real_candidate_count": r.real_candidate_count,
                        "split": r.split,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    splits_path = args.output_dir / "splits.json"
    split_summary = {
        "snapshot_name": manifest.get("snapshot_name"),
        "seed": args.seed,
        "train_ratio": TRAIN_RATIO,
        "val_ratio": VAL_RATIO,
        "message_ids_by_split": message_ids_by_split,
    }
    splits_path.write_text(
        json.dumps(split_summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    label_counts: dict[str, int] = defaultdict(int)
    negative_type_counts: dict[str, int] = defaultdict(int)
    split_pair_counts: dict[str, int] = defaultdict(int)
    split_label_counts: dict[str, dict[str, int]] = {
        s: {"positive": 0, "negative": 0} for s in ("train", "val", "test")
    }
    for r in records:
        label_counts["positive" if r.label == 1 else "negative"] += 1
        if r.label == 0:
            negative_type_counts[r.negative_type] += 1
        split_pair_counts[r.split] += 1
        split_label_counts[r.split]["positive" if r.label == 1 else "negative"] += 1

    summary = {
        "pairs_total": len(records),
        "labels": dict(label_counts),
        "negative_types": dict(negative_type_counts),
        "messages_total": len(messages),
        "split_message_counts": {s: len(ids) for s, ids in message_ids_by_split.items()},
        "split_pair_counts": dict(split_pair_counts),
        "split_label_counts": split_label_counts,
    }
    print(f"pairs -> {pairs_path}")
    print(f"splits -> {splits_path}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

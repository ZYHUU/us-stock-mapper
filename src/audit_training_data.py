from __future__ import annotations

import argparse
import json
import re
import sqlite3
from collections import Counter, defaultdict
from contextlib import closing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.database import PROJECT_ROOT
from src.mapper import Company, USStockMapper, load_enabled_companies

DEFAULT_SNAPSHOT_ROOT = PROJECT_ROOT / "data" / "snapshots"
DEFAULT_SNAPSHOT_NAME = "training-v1"
DEFAULT_REPORT_MD = PROJECT_ROOT / "reports" / "data_audit.md"
DEFAULT_REPORT_JSON = PROJECT_ROOT / "reports" / "data_audit.json"

CJK_PATTERN = re.compile(r"[一-鿿]")
TOP_N = 30


@dataclass
class LabeledMessage:
    message_id: int
    source_id: str
    source_type: str
    text: str
    correct_codes: list[str]
    confidence: str
    decision: str


def _readonly_connection(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    return connection


def resolve_snapshot(
    snapshot_root: Path, snapshot_name: str
) -> tuple[Path, Path, dict[str, Any]]:
    snapshot_dir = snapshot_root / snapshot_name
    manifest_path = snapshot_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"快照缺少 manifest.json：{manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    database_path = snapshot_dir / manifest["files"]["database"]["path"]
    companies_path = snapshot_dir / manifest["files"]["companies"]["path"]
    if not database_path.is_file():
        raise FileNotFoundError(f"快照数据库不存在：{database_path}")
    if not companies_path.is_file():
        raise FileNotFoundError(f"快照公司规则不存在：{companies_path}")
    return database_path, companies_path, manifest


def load_labeled_messages(connection: sqlite3.Connection) -> list[LabeledMessage]:
    rows = connection.execute(
        """
        WITH latest AS (
            SELECT a.*
            FROM annotations AS a
            WHERE a.id = (
                SELECT MAX(newer.id) FROM annotations AS newer
                WHERE newer.message_id = a.message_id
            )
        )
        SELECT
            m.id AS message_id, m.source_id, m.source_type, m.text,
            latest.correct_codes_json, latest.confidence, latest.decision
        FROM messages AS m
        JOIN latest ON latest.message_id = m.id
        WHERE TRIM(m.text) <> ''
        ORDER BY m.id
        """
    ).fetchall()
    messages: list[LabeledMessage] = []
    for row in rows:
        messages.append(
            LabeledMessage(
                message_id=int(row["message_id"]),
                source_id=str(row["source_id"] or ""),
                source_type=str(row["source_type"] or "unknown"),
                text=str(row["text"] or ""),
                correct_codes=list(dict.fromkeys(json.loads(row["correct_codes_json"]))),
                confidence=str(row["confidence"] or ""),
                decision=str(row["decision"] or ""),
            )
        )
    return messages


def load_upstream_codes_by_message(
    connection: sqlite3.Connection,
) -> dict[int, list[str]]:
    rows = connection.execute(
        """
        SELECT message_id, stocker_code
        FROM upstream_candidates
        WHERE TRIM(stocker_code) <> ''
        """
    ).fetchall()
    upstream: dict[int, list[str]] = defaultdict(list)
    for row in rows:
        upstream[int(row["message_id"])].append(str(row["stocker_code"]))
    return upstream


def load_target_tickers(connection: sqlite3.Connection) -> set[str]:
    rows = connection.execute(
        """
        SELECT ticker FROM securities
        WHERE mapper_candidate = 1 AND asset_type = 'stock'
        """
    ).fetchall()
    return {str(row["ticker"]) for row in rows}


def build_ticker_to_code(companies: list[Company]) -> dict[str, str]:
    return {company.ticker: company.canonical_code for company in companies}


def detect_language(text: str) -> str:
    return "cn" if CJK_PATTERN.search(text) else "en"


@dataclass
class AuditResult:
    manifest: dict[str, Any]
    counts: dict[str, int]
    source_type_distribution: Counter
    language_distribution: Counter
    confidence_distribution: Counter
    decision_distribution: Counter
    duplicate_text_groups: int
    duplicate_text_messages: int
    stale_codes: Counter
    company_positive_counts: dict[str, int]
    target_companies_total: int
    target_companies_with_positive: int
    target_companies_zero_positive: list[str]
    runtime_companies_total: int
    runtime_companies_with_positive: int
    recall_pairs_total: int
    recall_local_hits: int
    recall_upstream_hits: int
    recall_combined_hits: int
    missed_pairs: list[tuple[int, str, str]]
    false_positive_local: Counter
    false_positive_local_detail: Counter
    false_positive_upstream_only: Counter
    raw_pair_count: int
    raw_positive_pairs: int
    raw_negative_pairs: int
    dedup_pair_count: int
    dedup_positive_pairs: int
    dedup_negative_pairs: int


def run_audit(database_path: Path, companies_path: Path, manifest: dict[str, Any]) -> AuditResult:
    companies = load_enabled_companies(companies_path, database_path)
    code_to_name = {c.canonical_code: c.company_name for c in companies}
    ticker_to_code = build_ticker_to_code(companies)
    mapper = USStockMapper(companies)

    with closing(_readonly_connection(database_path)) as connection:
        messages = load_labeled_messages(connection)
        upstream_by_message = load_upstream_codes_by_message(connection)
        target_tickers = load_target_tickers(connection)

    target_codes = {ticker_to_code[t] for t in target_tickers if t in ticker_to_code}
    runtime_codes = set(code_to_name)

    positive_messages = [m for m in messages if m.correct_codes]
    counts = {
        "labeled_messages": len(messages),
        "positive_messages": len(positive_messages),
        "no_target_messages": len(messages) - len(positive_messages),
        "positive_tags_total": sum(len(m.correct_codes) for m in positive_messages),
    }

    source_type_distribution: Counter = Counter(m.source_type for m in messages)
    language_distribution: Counter = Counter(detect_language(m.text) for m in messages)
    confidence_distribution: Counter = Counter(m.confidence for m in messages)
    decision_distribution: Counter = Counter(m.decision for m in messages)

    text_groups: dict[str, list[int]] = defaultdict(list)
    for m in messages:
        text_groups[m.text].append(m.message_id)
    duplicate_groups = {text: ids for text, ids in text_groups.items() if len(ids) > 1}
    duplicate_text_groups = len(duplicate_groups)
    duplicate_text_messages = sum(len(ids) for ids in duplicate_groups.values())

    stale_codes: Counter = Counter()
    company_positive_counts: dict[str, int] = {code: 0 for code in target_codes}
    for m in positive_messages:
        for code in m.correct_codes:
            if code not in runtime_codes:
                stale_codes[code] += 1
            if code in target_codes:
                company_positive_counts[code] = company_positive_counts.get(code, 0) + 1

    zero_positive = sorted(code for code, n in company_positive_counts.items() if n == 0)
    runtime_positive_codes = {
        code for m in positive_messages for code in m.correct_codes if code in runtime_codes
    }

    recall_pairs_total = 0
    recall_local_hits = 0
    recall_upstream_hits = 0
    recall_combined_hits = 0
    missed_pairs: list[tuple[int, str, str]] = []

    false_positive_local: Counter = Counter()
    false_positive_local_detail: Counter = Counter()
    false_positive_upstream_only: Counter = Counter()

    raw_pair_count = 0
    raw_positive_pairs = 0
    raw_negative_pairs = 0

    representative_message_ids: set[int] = set()
    for text, ids in text_groups.items():
        representative_message_ids.add(min(ids))

    dedup_pair_count = 0
    dedup_positive_pairs = 0
    dedup_negative_pairs = 0

    by_id = {m.message_id: m for m in messages}

    for m in messages:
        local_matches = mapper.identify(m.text)
        local_codes = {match.canonical_code: match for match in local_matches}
        upstream_raw = upstream_by_message.get(m.message_id, [])
        upstream_codes = {
            ticker_to_code[t] for t in upstream_raw if t in ticker_to_code
        }
        correct = set(m.correct_codes)
        combined = set(local_codes) | upstream_codes

        for code in correct:
            recall_pairs_total += 1
            hit_local = code in local_codes
            hit_upstream = code in upstream_codes
            if hit_local:
                recall_local_hits += 1
            if hit_upstream:
                recall_upstream_hits += 1
            if hit_local or hit_upstream:
                recall_combined_hits += 1
            else:
                missed_pairs.append((m.message_id, m.source_id, code))

        false_positive_codes = combined - correct
        for code in false_positive_codes:
            match = local_codes.get(code)
            if match is not None:
                false_positive_local[code] += 1
                false_positive_local_detail[(code, match.mention, match.match_type)] += 1
            else:
                false_positive_upstream_only[code] += 1

        candidate_set = combined | correct
        pair_count = len(candidate_set)
        raw_pair_count += pair_count
        raw_positive_pairs += len(correct)
        raw_negative_pairs += pair_count - len(correct)

        if m.message_id in representative_message_ids:
            dedup_pair_count += pair_count
            dedup_positive_pairs += len(correct)
            dedup_negative_pairs += pair_count - len(correct)

    missed_pairs.sort(key=lambda item: item[2])

    return AuditResult(
        manifest=manifest,
        counts=counts,
        source_type_distribution=source_type_distribution,
        language_distribution=language_distribution,
        confidence_distribution=confidence_distribution,
        decision_distribution=decision_distribution,
        duplicate_text_groups=duplicate_text_groups,
        duplicate_text_messages=duplicate_text_messages,
        stale_codes=stale_codes,
        company_positive_counts=company_positive_counts,
        target_companies_total=len(target_codes),
        target_companies_with_positive=len(target_codes) - len(zero_positive),
        target_companies_zero_positive=[
            f"{code} ({code_to_name.get(code, '')})" for code in zero_positive
        ],
        runtime_companies_total=len(runtime_codes),
        runtime_companies_with_positive=len(runtime_positive_codes),
        recall_pairs_total=recall_pairs_total,
        recall_local_hits=recall_local_hits,
        recall_upstream_hits=recall_upstream_hits,
        recall_combined_hits=recall_combined_hits,
        missed_pairs=missed_pairs,
        false_positive_local=false_positive_local,
        false_positive_local_detail=false_positive_local_detail,
        false_positive_upstream_only=false_positive_upstream_only,
        raw_pair_count=raw_pair_count,
        raw_positive_pairs=raw_positive_pairs,
        raw_negative_pairs=raw_negative_pairs,
        dedup_pair_count=dedup_pair_count,
        dedup_positive_pairs=dedup_positive_pairs,
        dedup_negative_pairs=dedup_negative_pairs,
    )


def _pct(numerator: int, denominator: int) -> str:
    if denominator == 0:
        return "n/a"
    return f"{numerator / denominator * 100:.1f}%"


def to_json(result: AuditResult) -> dict[str, Any]:
    return {
        "snapshot": {
            "name": result.manifest.get("snapshot_name"),
            "cutoff_at": result.manifest.get("cutoff_at"),
        },
        "counts": result.counts,
        "distributions": {
            "source_type": dict(result.source_type_distribution),
            "language": dict(result.language_distribution),
            "confidence": dict(result.confidence_distribution),
            "decision": dict(result.decision_distribution),
        },
        "text_quality": {
            "duplicate_text_groups": result.duplicate_text_groups,
            "duplicate_text_messages": result.duplicate_text_messages,
            "stale_codes": dict(result.stale_codes),
        },
        "company_coverage": {
            "target_companies_total": result.target_companies_total,
            "target_companies_with_positive": result.target_companies_with_positive,
            "target_companies_zero_positive": result.target_companies_zero_positive,
            "runtime_companies_total": result.runtime_companies_total,
            "runtime_companies_with_positive": result.runtime_companies_with_positive,
            "positive_counts_by_code": result.company_positive_counts,
        },
        "candidate_recall": {
            "pairs_total": result.recall_pairs_total,
            "local_hits": result.recall_local_hits,
            "upstream_hits": result.recall_upstream_hits,
            "combined_hits": result.recall_combined_hits,
            "local_recall_rate": _pct(result.recall_local_hits, result.recall_pairs_total),
            "upstream_recall_rate": _pct(result.recall_upstream_hits, result.recall_pairs_total),
            "combined_recall_rate": _pct(result.recall_combined_hits, result.recall_pairs_total),
            "missed_pairs": [
                {"message_id": mid, "source_id": sid, "code": code}
                for mid, sid, code in result.missed_pairs
            ],
        },
        "false_positive_prone": {
            "local_by_code": result.false_positive_local.most_common(TOP_N),
            "local_by_code_and_mention": [
                {"code": code, "mention": mention, "match_type": match_type, "count": count}
                for (code, mention, match_type), count in (
                    result.false_positive_local_detail.most_common(TOP_N)
                )
            ],
            "upstream_only_by_code": result.false_positive_upstream_only.most_common(TOP_N),
        },
        "pair_counts": {
            "raw": {
                "total": result.raw_pair_count,
                "positive": result.raw_positive_pairs,
                "negative": result.raw_negative_pairs,
            },
            "dedup_exact_text": {
                "total": result.dedup_pair_count,
                "positive": result.dedup_positive_pairs,
                "negative": result.dedup_negative_pairs,
            },
        },
    }


def to_markdown(result: AuditResult) -> str:
    lines: list[str] = []
    lines.append("# 训练数据审计报告")
    lines.append("")
    lines.append(f"快照：`{result.manifest.get('snapshot_name')}`，固定时间：{result.manifest.get('cutoff_at')}")
    lines.append("")

    lines.append("## 基础统计")
    lines.append("")
    lines.append("| 指标 | 数量 |")
    lines.append("| --- | ---: |")
    lines.append(f"| 已标注消息 | {result.counts['labeled_messages']} |")
    lines.append(f"| 含目标公司正样本消息 | {result.counts['positive_messages']} |")
    lines.append(f"| 无目标公司消息 | {result.counts['no_target_messages']} |")
    lines.append(f"| 正确公司标签总数 | {result.counts['positive_tags_total']} |")
    lines.append("")

    lines.append("## 来源与语言分布")
    lines.append("")
    lines.append(f"来源类型：{dict(result.source_type_distribution)}")
    lines.append("")
    lines.append(f"语言（简单 CJK 判断）：{dict(result.language_distribution)}")
    lines.append("")
    lines.append(f"标注置信度：{dict(result.confidence_distribution)}")
    lines.append("")
    lines.append(f"标注决定类型：{dict(result.decision_distribution)}")
    lines.append("")

    lines.append("## 文本质量")
    lines.append("")
    lines.append(
        f"完全相同文本的重复组：{result.duplicate_text_groups} 组，"
        f"共 {result.duplicate_text_messages} 条消息（仅按精确文本去重，未做近似去重）"
    )
    if result.stale_codes:
        lines.append("")
        lines.append("发现停用/不存在的代码（不在当前 121 家运行时公司中）：")
        for code, count in result.stale_codes.most_common():
            lines.append(f"- `{code}`：{count} 次")
    else:
        lines.append("")
        lines.append("未发现停用或不存在的代码。")
    lines.append("")

    lines.append("## 公司覆盖率")
    lines.append("")
    lines.append(
        f"币安目标公司（`mapper_candidate=1`）共 {result.target_companies_total} 家，"
        f"其中 {result.target_companies_with_positive} 家至少有 1 条正样本"
        f"（{_pct(result.target_companies_with_positive, result.target_companies_total)}）。"
    )
    lines.append("")
    lines.append(
        f"运行时全部公司共 {result.runtime_companies_total} 家，"
        f"其中 {result.runtime_companies_with_positive} 家至少有 1 条正样本。"
    )
    if result.target_companies_zero_positive:
        lines.append("")
        lines.append(f"零正样本的目标公司（共 {len(result.target_companies_zero_positive)} 家）：")
        for entry in result.target_companies_zero_positive:
            lines.append(f"- {entry}")
    lines.append("")

    lines.append("### 正样本数量分布（按公司，仅目标公司，降序前 30）")
    lines.append("")
    lines.append("| 代码 | 正样本消息数 |")
    lines.append("| --- | ---: |")
    ranked = sorted(result.company_positive_counts.items(), key=lambda kv: kv[1], reverse=True)
    for code, count in ranked[:TOP_N]:
        lines.append(f"| {code} | {count} |")
    lines.append("")

    lines.append("## 候选召回率")
    lines.append("")
    lines.append(f"人工正确标签总数（作为召回分母）：{result.recall_pairs_total}")
    lines.append("")
    lines.append("| 候选来源 | 命中数 | 召回率 |")
    lines.append("| --- | ---: | ---: |")
    lines.append(
        f"| 本地规则 | {result.recall_local_hits} | "
        f"{_pct(result.recall_local_hits, result.recall_pairs_total)} |"
    )
    lines.append(
        f"| 上游 stocks/stoks | {result.recall_upstream_hits} | "
        f"{_pct(result.recall_upstream_hits, result.recall_pairs_total)} |"
    )
    lines.append(
        f"| 本地+上游合并 | {result.recall_combined_hits} | "
        f"{_pct(result.recall_combined_hits, result.recall_pairs_total)} |"
    )
    lines.append("")
    if result.missed_pairs:
        lines.append(f"两种候选都未召回的正确标签（共 {len(result.missed_pairs)} 条）：")
        for mid, sid, code in result.missed_pairs[:TOP_N]:
            lines.append(f"- message_id={mid} source_id={sid} code={code}")
        if len(result.missed_pairs) > TOP_N:
            lines.append(f"- ...（其余 {len(result.missed_pairs) - TOP_N} 条见 JSON 报告）")
    else:
        lines.append("没有被两种候选都漏掉的正确标签。")
    lines.append("")

    lines.append("## 最容易误报的代码（本地规则命中但不在人工正确标签中）")
    lines.append("")
    lines.append("| 代码 | 误报次数 |")
    lines.append("| --- | ---: |")
    for code, count in result.false_positive_local.most_common(TOP_N):
        lines.append(f"| {code} | {count} |")
    lines.append("")

    lines.append("### 按具体别名/命中词细分（代码, 命中词, 命中类型）")
    lines.append("")
    lines.append("| 代码 | 命中词 | 类型 | 误报次数 |")
    lines.append("| --- | --- | --- | ---: |")
    for (code, mention, match_type), count in result.false_positive_local_detail.most_common(TOP_N):
        lines.append(f"| {code} | {mention} | {match_type} | {count} |")
    lines.append("")

    if result.false_positive_upstream_only:
        lines.append("### 仅上游候选误报（本地规则未命中）")
        lines.append("")
        lines.append("| 代码 | 误报次数 |")
        lines.append("| --- | ---: |")
        for code, count in result.false_positive_upstream_only.most_common(TOP_N):
            lines.append(f"| {code} | {count} |")
        lines.append("")

    lines.append("## 可用训练文本对数量")
    lines.append("")
    lines.append("候选集合 = 本地规则命中 ∪ 上游候选 ∪ 人工正确标签（保证正样本一定在候选集合中）。")
    lines.append("")
    lines.append("| 统计口径 | 总对数 | 正样本对 | 负样本对 |")
    lines.append("| --- | ---: | ---: | ---: |")
    lines.append(
        f"| 原始（未去重） | {result.raw_pair_count} | {result.raw_positive_pairs} | "
        f"{result.raw_negative_pairs} |"
    )
    lines.append(
        f"| 精确文本去重后 | {result.dedup_pair_count} | {result.dedup_positive_pairs} | "
        f"{result.dedup_negative_pairs} |"
    )
    lines.append("")
    lines.append(
        "注：这里的去重只合并了完全相同的消息文本，未做近似重复检测（同一新闻不同转发措辞等）。"
        "近似去重留给 `build_training_dataset.py` 按第 7 节的切分约束统一处理。"
    )
    lines.append("")

    return "\n".join(lines)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="生成训练数据质量与覆盖率审计报告")
    parser.add_argument("--snapshot-root", type=Path, default=DEFAULT_SNAPSHOT_ROOT)
    parser.add_argument("--snapshot-name", default=DEFAULT_SNAPSHOT_NAME)
    parser.add_argument("--report-md", type=Path, default=DEFAULT_REPORT_MD)
    parser.add_argument("--report-json", type=Path, default=DEFAULT_REPORT_JSON)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    database_path, companies_path, manifest = resolve_snapshot(
        args.snapshot_root, args.snapshot_name
    )
    result = run_audit(database_path, companies_path, manifest)

    args.report_md.parent.mkdir(parents=True, exist_ok=True)
    args.report_md.write_text(to_markdown(result), encoding="utf-8", newline="\n")
    args.report_json.write_text(
        json.dumps(to_json(result), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"markdown -> {args.report_md}")
    print(f"json -> {args.report_json}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import sqlite3
import time
from contextlib import closing
from pathlib import Path

import joblib
import numpy as np
from scipy.sparse import csr_matrix, hstack

from src.audit_training_data import resolve_snapshot
from src.database import PROJECT_ROOT
from src.mapper import Company, USStockMapper, load_enabled_companies
from src.build_training_dataset import build_candidate_profile
from src.train_classic_model import PairExample, extract_structured_features

DEFAULT_SNAPSHOT_ROOT = PROJECT_ROOT / "data" / "snapshots"
DEFAULT_MODEL_PATH = (
    PROJECT_ROOT / "models" / "company_classifier" / "tfidf_structured__balanced" / "model.joblib"
)
DEFAULT_REPORT_PATH = PROJECT_ROOT / "reports" / "pipeline_latency.md"
DEFAULT_THRESHOLD = 0.8179462006201739
DEFAULT_SAMPLE_SIZE = 2000
WARMUP_SIZE = 10


def _readonly_connection(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    return connection


def run_pipeline(
    text: str,
    source_type: str,
    mapper: USStockMapper,
    code_to_company: dict[str, Company],
    word_vectorizer,
    char_vectorizer,
    classifier,
    use_structured: bool,
    threshold: float,
    svd=None,
    upstream_codes: set[str] | None = None,
) -> dict:
    # 第1步：候选召回。mapper.companies 已经在启动时从 SQLite 加载好，
    # identify() 返回的 canonical_code 本身就是标准代码，这一步同时完成了
    # “规则召回”和“映射到标准股票代码”，之后不需要再查一次数据库。
    matches = mapper.identify(text)
    if not matches:
        return {"n_candidates": 0, "predicted_codes": []}

    # 候选数量特征必须是"规则+上游"的真实并集大小，和训练时的算法保持一致
    # （不能只数本地规则命中数，否则会复现训练/线上不一致的坑，见本轮踩坑记录）。
    real_candidate_count = len(
        {match.canonical_code for match in matches} | (upstream_codes or set())
    )

    # 第2步：文本预处理——把每个候选拼成“消息+候选公司资料”的文本对
    pair_texts = []
    structured_rows = []
    codes = []
    for match in matches:
        company = code_to_company[match.canonical_code]
        profile = build_candidate_profile(company)
        pair_texts.append(f"{text}\n[CANDIDATE]\n{profile}")
        codes.append(match.canonical_code)
        if use_structured:
            example = PairExample(
                message_id=0, source_type=source_type, message_text=text,
                candidate_code=match.canonical_code, candidate_profile=profile,
                label=0, negative_type="", split="",
            )
            structured_rows.append(
                extract_structured_features(example, company, real_candidate_count)
            )

    word_matrix = word_vectorizer.transform(pair_texts)
    char_matrix = char_vectorizer.transform(pair_texts)

    if svd is not None:
        # LightGBM 版本：TF-IDF 先降维，再和结构化特征拼成稠密矩阵
        tfidf = hstack([word_matrix, char_matrix]).tocsr()
        reduced = svd.transform(tfidf)
        structured = np.asarray(structured_rows, dtype=np.float64)
        X = np.hstack([reduced, structured])
    else:
        parts = [word_matrix, char_matrix]
        if use_structured:
            parts.append(csr_matrix(np.asarray(structured_rows, dtype=np.float64)))
        X = hstack(parts).tocsr()

    # 第3步：模型推理
    scores = classifier.predict_proba(X)[:, 1]

    # 第4步：阈值过滤，得到最终标准代码（已经是 SQLite 里的 canonical_code）
    predicted_codes = [code for code, score in zip(codes, scores) if score >= threshold]
    return {"n_candidates": len(matches), "predicted_codes": predicted_codes}


def percentile(values: list[float], p: float) -> float | None:
    return float(np.percentile(np.array(values), p)) if values else None


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="测量完整链路（候选召回+预处理+模型推理+代码映射）的端到端延迟"
    )
    parser.add_argument("--snapshot-root", type=Path, default=DEFAULT_SNAPSHOT_ROOT)
    parser.add_argument("--snapshot-name", required=True)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    database_path, companies_path, _manifest = resolve_snapshot(
        args.snapshot_root, args.snapshot_name
    )
    # 公司列表在“启动时”加载一次，和真实服务的做法一致，不计入单条消息的延迟
    companies = load_enabled_companies(companies_path, database_path)
    mapper = USStockMapper(companies)
    code_to_company = {c.canonical_code: c for c in companies}
    ticker_to_code = {c.ticker: c.canonical_code for c in companies}

    bundle = joblib.load(args.model_path)
    word_vectorizer = bundle["word_vectorizer"]
    char_vectorizer = bundle["char_vectorizer"]
    classifier = bundle["classifier"]
    svd = bundle.get("svd")
    use_structured = bundle.get("use_structured", False) or svd is not None

    with closing(_readonly_connection(database_path)) as connection:
        rows = connection.execute(
            "SELECT text, source_type, upstream_candidates_json FROM messages "
            "WHERE TRIM(text) <> '' ORDER BY RANDOM() LIMIT ?",
            (args.sample_size,),
        ).fetchall()
    messages = []
    for r in rows:
        upstream_raw = json.loads(r["upstream_candidates_json"] or "[]")
        upstream_codes = {
            ticker_to_code[str(c.get("stocker_code") or "")]
            for c in upstream_raw
            if str(c.get("stocker_code") or "") in ticker_to_code
        }
        messages.append(
            (str(r["text"]), str(r["source_type"] or "unknown"), upstream_codes)
        )

    for text, source_type, upstream_codes in messages[:WARMUP_SIZE]:
        run_pipeline(
            text, source_type, mapper, code_to_company, word_vectorizer, char_vectorizer,
            classifier, use_structured, args.threshold, svd=svd, upstream_codes=upstream_codes,
        )

    no_candidate_latencies_ms: list[float] = []
    with_candidate_latencies_ms: list[float] = []
    single_candidate_latencies_ms: list[float] = []
    multi_candidate_latencies_ms: list[float] = []
    candidate_counts: list[int] = []

    for text, source_type, upstream_codes in messages:
        start = time.perf_counter()
        result = run_pipeline(
            text, source_type, mapper, code_to_company, word_vectorizer, char_vectorizer,
            classifier, use_structured, args.threshold, svd=svd, upstream_codes=upstream_codes,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000
        n = result["n_candidates"]
        candidate_counts.append(n)
        if n == 0:
            no_candidate_latencies_ms.append(elapsed_ms)
        else:
            with_candidate_latencies_ms.append(elapsed_ms)
            if n == 1:
                single_candidate_latencies_ms.append(elapsed_ms)
            else:
                multi_candidate_latencies_ms.append(elapsed_ms)

    summary = {
        "sample_size": len(messages),
        "no_candidate": {
            "count": len(no_candidate_latencies_ms),
            "p50_ms": percentile(no_candidate_latencies_ms, 50),
            "p95_ms": percentile(no_candidate_latencies_ms, 95),
            "p99_ms": percentile(no_candidate_latencies_ms, 99),
        },
        "with_candidate_any": {
            "count": len(with_candidate_latencies_ms),
            "p50_ms": percentile(with_candidate_latencies_ms, 50),
            "p95_ms": percentile(with_candidate_latencies_ms, 95),
            "p99_ms": percentile(with_candidate_latencies_ms, 99),
        },
        "single_candidate": {
            "count": len(single_candidate_latencies_ms),
            "p50_ms": percentile(single_candidate_latencies_ms, 50),
            "p95_ms": percentile(single_candidate_latencies_ms, 95),
            "p99_ms": percentile(single_candidate_latencies_ms, 99),
        },
        "multi_candidate": {
            "count": len(multi_candidate_latencies_ms),
            "p50_ms": percentile(multi_candidate_latencies_ms, 50),
            "p95_ms": percentile(multi_candidate_latencies_ms, 95),
            "p99_ms": percentile(multi_candidate_latencies_ms, 99),
            "max_candidates_seen": max(candidate_counts) if candidate_counts else 0,
        },
    }

    def _fmt_ms(v: float | None) -> str:
        return "n/a" if v is None else f"{v:.2f}ms"

    lines = [
        "# 完整链路端到端延迟（本机测得，仅供参考，不是生产环境标定值）",
        "",
        f"模型：`{args.model_path}`，阈值：{args.threshold:.4f}，随机抽样 {summary['sample_size']} 条消息"
        "（覆盖真实分布：绝大多数消息没有候选公司，只有少数会进入模型推理）。",
        "",
        "链路 = 候选召回（本地规则）→ 文本预处理（TF-IDF向量化）→ 模型推理（LogisticRegression）"
        "→ 阈值过滤映射标准代码。候选公司列表在启动时从 SQLite 加载一次，单条消息推理不再查库。",
        "",
        "| 场景 | 消息数 | p50 | p95 | p99 |",
        "| --- | ---: | ---: | ---: | ---: |",
        f"| 无候选（只有候选召回，跳过模型） | {summary['no_candidate']['count']} | "
        f"{_fmt_ms(summary['no_candidate']['p50_ms'])} | {_fmt_ms(summary['no_candidate']['p95_ms'])} | "
        f"{_fmt_ms(summary['no_candidate']['p99_ms'])} |",
        f"| 有候选（含模型推理，任意候选数） | {summary['with_candidate_any']['count']} | "
        f"{_fmt_ms(summary['with_candidate_any']['p50_ms'])} | {_fmt_ms(summary['with_candidate_any']['p95_ms'])} | "
        f"{_fmt_ms(summary['with_candidate_any']['p99_ms'])} |",
        f"| 恰好1个候选 | {summary['single_candidate']['count']} | "
        f"{_fmt_ms(summary['single_candidate']['p50_ms'])} | {_fmt_ms(summary['single_candidate']['p95_ms'])} | "
        f"{_fmt_ms(summary['single_candidate']['p99_ms'])} |",
        f"| 多个候选（2个及以上，最多见过{summary['multi_candidate']['max_candidates_seen']}个） | "
        f"{summary['multi_candidate']['count']} | {_fmt_ms(summary['multi_candidate']['p50_ms'])} | "
        f"{_fmt_ms(summary['multi_candidate']['p95_ms'])} | {_fmt_ms(summary['multi_candidate']['p99_ms'])} |",
        "",
        "注：本机CPU、单线程、未做任何工程优化（没有ONNX、没有批量攒批、没有连接池预热）下测得，"
        "生产环境实际数值会因硬件、并发方式不同而变化，这里主要是判断这套架构的量级是否可能进入100ms预算，"
        "不是最终SLA承诺。",
    ]
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text("\n".join(lines), encoding="utf-8", newline="\n")

    print(f"report -> {args.report_path}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from src.audit_training_data import resolve_snapshot
from src.database import PROJECT_ROOT
from src.mapper import Company, _contains, _contains_ticker, load_enabled_companies

DEFAULT_PAIRS_PATH = PROJECT_ROOT / "data" / "training" / "candidate_pairs.jsonl"
DEFAULT_SNAPSHOT_ROOT = PROJECT_ROOT / "data" / "snapshots"
DEFAULT_MODEL_ROOT = PROJECT_ROOT / "models" / "company_classifier"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "reports" / "classic_model_val_report.md"

# 与训练规划第6节列出的上下文词一致，只是同时保留中英文两套写法。
CONTEXT_KEYWORDS = [
    "stock", "share", "shares", "company", "earnings", "revenue", "CEO", "IPO",
    "股票", "公司", "财报", "营收", "股价", "收购", "上市", "季度",
]

WORD_NGRAM_RANGE = (1, 2)
CHAR_NGRAM_RANGE = (2, 5)
# sklearn 默认 token_pattern（r"(?u)\b\w\w+\b"）不切分中文——一整句没有空格的
# 中文会被 \w+ 当成一个"词"整体吃掉，词表里会出现接近完整句子的原文片段
# （抓取自第三方社交媒体/新闻的训练数据，公开发布模型文件会把这些片段带出去，
# 隐私检查时发现，2026-08-24）。改成只匹配 ASCII 单词或单个中文字符，中文的
# 泛化信号完全交给 char n-gram 承担（char_vectorizer 按字符切，不受此影响）。
WORD_TOKEN_PATTERN = r"[A-Za-z0-9]+|[一-鿿]"
CLASS_WEIGHT_VARIANTS = {"balanced": "balanced", "unweighted": None}
FEATURE_VARIANTS = ("tfidf", "tfidf_structured")

STRUCTURED_FEATURE_NAMES = [
    "has_dollar_ticker",
    "has_bare_ticker",
    "has_alias",
    "shortest_hit_len",
    "num_alias_hits",
    "has_negative_context",
    "context_present",
    "ticker_len",
    "alias_count_total",
    "message_len",
    "is_news",
    "candidates_in_message",
]


@dataclass
class PairExample:
    message_id: int
    source_type: str
    message_text: str
    candidate_code: str
    candidate_profile: str
    label: int
    negative_type: str
    split: str
    real_candidate_count: int = 1


def load_pairs(path: Path) -> list[PairExample]:
    examples: list[PairExample] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            row = json.loads(line)
            examples.append(
                PairExample(
                    message_id=int(row["message_id"]),
                    source_type=str(row["source_type"]),
                    message_text=str(row["message_text"]),
                    candidate_code=str(row["candidate_code"]),
                    candidate_profile=str(row["candidate_profile"]),
                    label=int(row["label"]),
                    negative_type=str(row["negative_type"]),
                    split=str(row["split"]),
                    real_candidate_count=int(row.get("real_candidate_count", 1)),
                )
            )
    return examples


def pair_text(example: PairExample) -> str:
    # 输入按第4节的“消息—候选公司文本对”格式拼接，保证同一消息的不同候选
    # 得到不同的文本对（否则TF-IDF会把它们向量化成完全相同的行）。
    return f"{example.message_text}\n[CANDIDATE]\n{example.candidate_profile}"


def extract_structured_features(
    example: PairExample, company: Company, real_candidate_count: int
) -> list[float]:
    text = re.sub(r"https?://\S+", " ", example.message_text)
    alias_terms = list(company.aliases) + list(company.brands)
    alias_hits = [keyword for keyword in alias_terms if _contains(text, keyword)]

    return [
        1.0 if re.search(rf"\${re.escape(company.ticker)}(?![A-Za-z0-9])", text) else 0.0,
        1.0 if _contains_ticker(text, company.ticker) else 0.0,
        1.0 if alias_hits else 0.0,
        float(min((len(hit) for hit in alias_hits), default=0)),
        float(len(alias_hits)),
        1.0 if any(_contains(text, kw) for kw in company.negative_contexts) else 0.0,
        1.0 if any(_contains(text, kw) for kw in CONTEXT_KEYWORDS) else 0.0,
        float(len(company.ticker)),
        float(len(alias_terms)),
        float(len(text)),
        1.0 if example.source_type == "news" else 0.0,
        float(real_candidate_count),
    ]


def build_structured_matrix(
    examples: list[PairExample],
    code_to_company: dict[str, Company],
) -> np.ndarray:
    # real_candidate_count 直接来自 candidate_pairs.jsonl（build_training_dataset.py
    # 在填充 random_topup/background 负样本之前就记录好了），
    # 和线上推理时“规则+上游候选的真实并集大小”算法保持一致，
    # 不能再用 Counter(message_id) 数训练文件里的pair行数——那个数字被负样本填充污染了。
    rows = [
        extract_structured_features(
            example, code_to_company[example.candidate_code], example.real_candidate_count
        )
        for example in examples
    ]
    return np.asarray(rows, dtype=np.float64)


def build_feature_matrix(
    texts: list[str],
    word_vectorizer: TfidfVectorizer,
    char_vectorizer: TfidfVectorizer,
    fit: bool,
    structured: np.ndarray | None,
):
    if fit:
        word_matrix = word_vectorizer.fit_transform(texts)
        char_matrix = char_vectorizer.fit_transform(texts)
    else:
        word_matrix = word_vectorizer.transform(texts)
        char_matrix = char_vectorizer.transform(texts)
    parts = [word_matrix, char_matrix]
    if structured is not None:
        parts.append(csr_matrix(structured))
    return hstack(parts).tocsr()


def _current_git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def write_model_meta(
    variant_dir: Path,
    variant: str,
    snapshot_name: str,
    metrics: dict,
    model_version: str | None = None,
) -> None:
    """在 model.joblib 旁边写一份 meta.json，让模型文件自带“身份证”：
    训练时间、用的数据快照、验证集指标、当时的 git commit，以及可选的显式版本号
    （对应 semantic_matcher.py 里手写的 LR_MODEL_VERSION/LIGHTGBM_MODEL_VERSION）。
    这样重训时即使忘了手动改版本号，也能靠 trained_at/git_commit/snapshot_name 追溯到
    是哪一次训练产出的模型，不会跟旧模型混淆。"""
    meta = {
        "variant": variant,
        "model_version": model_version,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "snapshot_name": snapshot_name,
        "git_commit": _current_git_commit(),
        "val_metrics": {
            key: value
            for key, value in metrics.items()
            if key in {"threshold", "precision", "recall", "f1", "tp", "fp", "fn", "tn"}
        },
    }
    (variant_dir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def evaluate_at_threshold(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    y_pred = (scores >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="训练 TF-IDF(word+char) + LogisticRegression 经典基线，并做结构化特征消融"
    )
    parser.add_argument("--snapshot-root", type=Path, default=DEFAULT_SNAPSHOT_ROOT)
    parser.add_argument("--snapshot-name", required=True, help="生成 candidate_pairs.jsonl 时使用的快照名称")
    parser.add_argument("--pairs-path", type=Path, default=DEFAULT_PAIRS_PATH)
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument(
        "--evaluate-test",
        action="store_true",
        help="同时在冻结测试集上评估（测试集在正式定稿前应保持冻结，默认不启用）",
    )
    parser.add_argument(
        "--model-version",
        help=(
            "写入 meta.json 的显式版本号（例如 classic-lr-v4），"
            "对应 semantic_matcher.py 里的 LR_MODEL_VERSION；不传则 meta.json 里留空，"
            "只记录 trained_at/git_commit/snapshot_name 用于追溯"
        ),
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    database_path, companies_path, _manifest = resolve_snapshot(
        args.snapshot_root, args.snapshot_name
    )
    companies = load_enabled_companies(companies_path, database_path)
    code_to_company = {c.canonical_code: c for c in companies}

    examples = load_pairs(args.pairs_path)

    train_examples = [e for e in examples if e.split == "train"]
    val_examples = [e for e in examples if e.split == "val"]
    test_examples = [e for e in examples if e.split == "test"] if args.evaluate_test else []

    unknown_codes = {
        e.candidate_code for e in examples if e.candidate_code not in code_to_company
    }
    if unknown_codes:
        raise SystemExit(
            f"候选代码在当前快照公司列表中找不到，快照可能与 candidate_pairs.jsonl 不匹配: {unknown_codes}"
        )

    y_train = np.array([e.label for e in train_examples])
    y_val = np.array([e.label for e in val_examples])

    train_texts = [pair_text(e) for e in train_examples]
    val_texts = [pair_text(e) for e in val_examples]

    train_structured = build_structured_matrix(train_examples, code_to_company)
    val_structured = build_structured_matrix(val_examples, code_to_company)

    args.model_root.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []

    for feature_variant in FEATURE_VARIANTS:
        use_structured = feature_variant == "tfidf_structured"

        word_vectorizer = TfidfVectorizer(
            ngram_range=WORD_NGRAM_RANGE, lowercase=True, min_df=2, max_features=20000,
            token_pattern=WORD_TOKEN_PATTERN,
        )
        char_vectorizer = TfidfVectorizer(
            analyzer="char", ngram_range=CHAR_NGRAM_RANGE, lowercase=False, min_df=2, max_features=20000
        )

        X_train = build_feature_matrix(
            train_texts, word_vectorizer, char_vectorizer, fit=True,
            structured=train_structured if use_structured else None,
        )
        X_val = build_feature_matrix(
            val_texts, word_vectorizer, char_vectorizer, fit=False,
            structured=val_structured if use_structured else None,
        )

        for weight_label, class_weight in CLASS_WEIGHT_VARIANTS.items():
            variant_name = f"{feature_variant}__{weight_label}"
            classifier = LogisticRegression(
                class_weight=class_weight, max_iter=5000, solver="liblinear"
            )
            classifier.fit(X_train, y_train)

            val_scores = classifier.predict_proba(X_val)[:, 1]
            metrics = evaluate_at_threshold(y_val, val_scores, threshold=0.5)
            metrics["variant"] = variant_name

            variant_dir = args.model_root / variant_name
            variant_dir.mkdir(parents=True, exist_ok=True)
            joblib.dump(
                {
                    "word_vectorizer": word_vectorizer,
                    "char_vectorizer": char_vectorizer,
                    "classifier": classifier,
                    "use_structured": use_structured,
                    "structured_feature_names": STRUCTURED_FEATURE_NAMES if use_structured else [],
                },
                variant_dir / "model.joblib",
            )
            write_model_meta(
                variant_dir,
                variant_name,
                args.snapshot_name,
                metrics,
                model_version=args.model_version,
            )
            with (variant_dir / "predictions_val.jsonl").open(
                "w", encoding="utf-8", newline="\n"
            ) as file:
                for example, score in zip(val_examples, val_scores):
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

            if args.evaluate_test:
                test_texts = [pair_text(e) for e in test_examples]
                test_structured = build_structured_matrix(test_examples, code_to_company)
                X_test = build_feature_matrix(
                    test_texts, word_vectorizer, char_vectorizer, fit=False,
                    structured=test_structured if use_structured else None,
                )
                y_test = np.array([e.label for e in test_examples])
                test_scores = classifier.predict_proba(X_test)[:, 1]
                metrics["test"] = evaluate_at_threshold(y_test, test_scores, threshold=0.5)

            results.append(metrics)
            print(json.dumps(metrics, ensure_ascii=False))

    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# 经典模型基线（验证集，阈值=0.5，尚未做阈值搜索）", ""]
    lines.append(f"训练集 {len(train_examples)} 对，验证集 {len(val_examples)} 对。")
    if args.evaluate_test:
        lines.append(
            f"警告：本次运行同时评估了测试集（{len(test_examples)} 对）——"
            "这会消耗测试集的“冻结”状态，仅应在准备做最终定稿评估时使用。"
        )
    lines.append("")
    lines.append("| 变体 | Precision | Recall | F1 | TP | FP | FN | TN |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for m in results:
        lines.append(
            f"| {m['variant']} | {m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} | "
            f"{m['tp']} | {m['fp']} | {m['fn']} | {m['tn']} |"
        )
    lines.append("")
    lines.append(
        "注：这里的 Precision/Recall/F1 是在固定阈值 0.5 下的粗略结果，"
        "仅用于快速比较四个变体。正式阈值搜索留给 `evaluate_model.py` 在验证集上完成。"
    )
    args.report_path.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    print(f"report -> {args.report_path}")


if __name__ == "__main__":
    main()

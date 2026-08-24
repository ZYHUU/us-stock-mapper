from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from lightgbm import LGBMClassifier
from scipy.sparse import hstack
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from src.audit_training_data import resolve_snapshot
from src.database import PROJECT_ROOT
from src.mapper import load_enabled_companies
from src.train_classic_model import (
    CHAR_NGRAM_RANGE,
    CLASS_WEIGHT_VARIANTS,
    STRUCTURED_FEATURE_NAMES,
    WORD_NGRAM_RANGE,
    WORD_TOKEN_PATTERN,
    build_structured_matrix,
    evaluate_at_threshold,
    load_pairs,
    pair_text,
    write_model_meta,
)

DEFAULT_PAIRS_PATH = PROJECT_ROOT / "data" / "training" / "candidate_pairs.jsonl"
DEFAULT_SNAPSHOT_ROOT = PROJECT_ROOT / "data" / "snapshots"
DEFAULT_MODEL_ROOT = PROJECT_ROOT / "models" / "company_classifier"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "reports" / "lightgbm_val_report.md"

SVD_COMPONENTS = 100
RANDOM_SEED = 20260821


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="训练 LightGBM（TF-IDF降维 + 结构化特征）作为经典ML的第二个对照模型"
    )
    parser.add_argument("--snapshot-root", type=Path, default=DEFAULT_SNAPSHOT_ROOT)
    parser.add_argument("--snapshot-name", required=True)
    parser.add_argument("--pairs-path", type=Path, default=DEFAULT_PAIRS_PATH)
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--svd-components", type=int, default=SVD_COMPONENTS)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument(
        "--model-version",
        help=(
            "写入 meta.json 的显式版本号（例如 lightgbm-shadow-v3），"
            "对应 semantic_matcher.py 里的 LIGHTGBM_MODEL_VERSION；不传则 meta.json 里留空"
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

    unknown_codes = {
        e.candidate_code for e in examples if e.candidate_code not in code_to_company
    }
    if unknown_codes:
        raise SystemExit(f"候选代码与当前快照公司列表不匹配: {unknown_codes}")

    train_examples = [e for e in examples if e.split == "train"]
    val_examples = [e for e in examples if e.split == "val"]

    y_train = np.array([e.label for e in train_examples])
    y_val = np.array([e.label for e in val_examples])

    train_texts = [pair_text(e) for e in train_examples]
    val_texts = [pair_text(e) for e in val_examples]

    word_vectorizer = TfidfVectorizer(
        ngram_range=WORD_NGRAM_RANGE, lowercase=True, min_df=2, max_features=20000,
        token_pattern=WORD_TOKEN_PATTERN,
    )
    char_vectorizer = TfidfVectorizer(
        analyzer="char", ngram_range=CHAR_NGRAM_RANGE, lowercase=False, min_df=2, max_features=20000
    )
    train_tfidf = hstack(
        [word_vectorizer.fit_transform(train_texts), char_vectorizer.fit_transform(train_texts)]
    ).tocsr()
    val_tfidf = hstack(
        [word_vectorizer.transform(val_texts), char_vectorizer.transform(val_texts)]
    ).tocsr()

    n_components = min(args.svd_components, min(train_tfidf.shape) - 1)
    svd = TruncatedSVD(n_components=n_components, random_state=args.seed)
    train_svd = svd.fit_transform(train_tfidf)
    val_svd = svd.transform(val_tfidf)

    train_structured = build_structured_matrix(train_examples, code_to_company)
    val_structured = build_structured_matrix(val_examples, code_to_company)

    X_train = np.hstack([train_svd, train_structured])
    X_val = np.hstack([val_svd, val_structured])

    feature_names = [f"svd_{i}" for i in range(n_components)] + STRUCTURED_FEATURE_NAMES

    args.model_root.mkdir(parents=True, exist_ok=True)
    results = []
    for weight_label, class_weight in CLASS_WEIGHT_VARIANTS.items():
        variant_name = f"lgbm_svd_structured__{weight_label}"
        scale_pos_weight = (
            (len(y_train) - y_train.sum()) / max(y_train.sum(), 1)
            if class_weight == "balanced"
            else 1.0
        )
        classifier = LGBMClassifier(
            n_estimators=300,
            num_leaves=31,
            learning_rate=0.05,
            random_state=args.seed,
            scale_pos_weight=scale_pos_weight,
            verbosity=-1,
        )
        classifier.fit(X_train, y_train)

        val_scores = classifier.predict_proba(X_val)[:, 1]
        metrics = evaluate_at_threshold(y_val, val_scores, threshold=0.5)
        metrics["variant"] = variant_name

        variant_dir = args.model_root / variant_name
        variant_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model_type": "lightgbm_svd_structured",
                "word_vectorizer": word_vectorizer,
                "char_vectorizer": char_vectorizer,
                "svd": svd,
                "classifier": classifier,
                "structured_feature_names": STRUCTURED_FEATURE_NAMES,
                "feature_names": feature_names,
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

        results.append(metrics)
        print(json.dumps(metrics, ensure_ascii=False))

    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# LightGBM 基线（TF-IDF SVD降维 + 结构化特征，验证集，阈值=0.5）",
        "",
        f"SVD维度: {n_components}，训练集 {len(train_examples)} 对，验证集 {len(val_examples)} 对。",
        "",
        "| 变体 | Precision | Recall | F1 | TP | FP | FN | TN |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for m in results:
        lines.append(
            f"| {m['variant']} | {m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} | "
            f"{m['tp']} | {m['fp']} | {m['fn']} | {m['tn']} |"
        )
    lines.append("")
    lines.append(
        "这里只是阈值0.5下的粗略对比；正式阈值搜索与和其他变体的统一比较见 `evaluate_model.py` "
        "生成的 `reports/model_evaluation.md`（重新运行该脚本即可自动纳入这两个 LightGBM 变体）。"
    )
    args.report_path.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    print(f"report -> {args.report_path}")


if __name__ == "__main__":
    main()

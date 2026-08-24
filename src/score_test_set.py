from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from scipy.sparse import csr_matrix, hstack

from src.audit_training_data import resolve_snapshot
from src.database import PROJECT_ROOT
from src.mapper import load_enabled_companies
from src.train_classic_model import build_structured_matrix, load_pairs, pair_text

DEFAULT_PAIRS_PATH = PROJECT_ROOT / "data" / "training" / "candidate_pairs.jsonl"
DEFAULT_SNAPSHOT_ROOT = PROJECT_ROOT / "data" / "snapshots"
DEFAULT_MODEL_ROOT = PROJECT_ROOT / "models" / "company_classifier"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="用已保存的 sklearn/LightGBM 模型 bundle 对冻结测试集打分一次，不重新训练、不重新搜索阈值"
    )
    parser.add_argument("--snapshot-root", type=Path, default=DEFAULT_SNAPSHOT_ROOT)
    parser.add_argument("--snapshot-name", required=True)
    parser.add_argument("--pairs-path", type=Path, default=DEFAULT_PAIRS_PATH)
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument(
        "--variants", nargs="*", default=None, help="只处理这些变体目录名，默认处理所有含 model.joblib 的目录"
    )
    parser.add_argument(
        "--confirm-test",
        action="store_true",
        help="必须显式传入，确认要在冻结测试集上打分（一次性操作）",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if not args.confirm_test:
        raise SystemExit("测试集应保持冻结，如确实要打分请显式加 --confirm-test")

    database_path, companies_path, _manifest = resolve_snapshot(
        args.snapshot_root, args.snapshot_name
    )
    companies = load_enabled_companies(companies_path, database_path)
    code_to_company = {c.canonical_code: c for c in companies}

    all_examples = load_pairs(args.pairs_path)
    examples = [e for e in all_examples if e.split == "test"]
    if not examples:
        raise SystemExit("candidate_pairs.jsonl 里没有 split=test 的样本")

    texts = [pair_text(e) for e in examples]
    structured = build_structured_matrix(examples, code_to_company)

    variant_dirs = sorted(p.parent for p in args.model_root.glob("*/model.joblib"))
    if args.variants:
        variant_dirs = [d for d in variant_dirs if d.name in args.variants]

    for variant_dir in variant_dirs:
        bundle = joblib.load(variant_dir / "model.joblib")
        word_matrix = bundle["word_vectorizer"].transform(texts)
        char_matrix = bundle["char_vectorizer"].transform(texts)

        if "svd" in bundle:
            tfidf = hstack([word_matrix, char_matrix]).tocsr()
            reduced = bundle["svd"].transform(tfidf)
            X = np.hstack([reduced, structured])
        else:
            parts = [word_matrix, char_matrix]
            if bundle.get("use_structured"):
                parts.append(csr_matrix(structured))
            X = hstack(parts).tocsr()

        scores = bundle["classifier"].predict_proba(X)[:, 1]
        out_path = variant_dir / "predictions_test.jsonl"
        with out_path.open("w", encoding="utf-8", newline="\n") as file:
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
        print(f"{variant_dir.name}: wrote {len(examples)} predictions -> {out_path}")


if __name__ == "__main__":
    main()

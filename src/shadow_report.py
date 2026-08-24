from __future__ import annotations

import argparse
import json
from contextlib import closing
from pathlib import Path

from src.database import DEFAULT_DB_PATH, connect, initialize


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="汇总影子运行结果：整体一致率 + 拉出规则与模型有分歧的消息供人工复核"
    )
    parser.add_argument("--database", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument(
        "--model-version",
        default=None,
        help="只看某一个模型版本；不传则汇总 shadow_predictions 表里出现过的所有版本",
    )
    parser.add_argument("--limit", type=int, default=20, help="每个模型版本最多列出多少条分歧消息")
    return parser


def report_one(connection, model_version: str, limit: int) -> None:
    summary_row = connection.execute(
        """
        SELECT
            COUNT(*) AS total,
            SUM(CASE WHEN agrees = 1 THEN 1 ELSE 0 END) AS agrees
        FROM shadow_predictions
        WHERE model_version = ?
        """,
        (model_version,),
    ).fetchone()
    total = int(summary_row["total"] or 0)
    agrees = int(summary_row["agrees"] or 0)

    disagreements = connection.execute(
        """
        SELECT
            sp.id, sp.message_id, sp.rule_codes_json, sp.model_codes_json,
            sp.candidates_json, sp.created_at,
            m.source_id, m.text
        FROM shadow_predictions AS sp
        JOIN messages AS m ON m.id = sp.message_id
        WHERE sp.model_version = ? AND sp.agrees = 0
        ORDER BY sp.id DESC
        LIMIT ?
        """,
        (model_version, limit),
    ).fetchall()

    print(
        f"\n=== 模型版本={model_version} ===\n"
        f"共 {total} 条，一致 {agrees} 条（{(agrees / total * 100) if total else 0:.1f}%），"
        f"分歧 {total - agrees} 条"
    )
    print(f"最近 {len(disagreements)} 条分歧（规则 vs 模型）：")
    for row in disagreements:
        rule_codes = json.loads(row["rule_codes_json"])
        model_codes = json.loads(row["model_codes_json"])
        text_preview = row["text"][:80].replace("\n", " ")
        print(
            f"- message_id={row['message_id']} source_id={row['source_id']} "
            f"规则={rule_codes} 模型={model_codes}"
        )
        print(f"    {text_preview}")


def main() -> None:
    args = _build_parser().parse_args()
    initialize(args.database)
    with closing(connect(args.database)) as connection:
        if args.model_version:
            versions = [args.model_version]
        else:
            rows = connection.execute(
                "SELECT DISTINCT model_version FROM shadow_predictions ORDER BY model_version"
            ).fetchall()
            versions = [row["model_version"] for row in rows]
            if not versions:
                print("shadow_predictions 表目前是空的（影子运行还没有产生数据）")
                return

        for model_version in versions:
            report_one(connection, model_version, args.limit)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import csv
from contextlib import closing
from datetime import datetime, timedelta, timezone
from pathlib import Path

from src.database import DEFAULT_DB_PATH, connect, initialize, save_message
from src.mapper import default_mapper

MODEL_VERSION = "rules-v2"


def _earliest_existing_news_time(database: Path) -> datetime | None:
    initialize(database)
    with closing(connect(database)) as connection:
        row = connection.execute(
            """
            SELECT MIN(COALESCE(NULLIF(published_at, ''), received_at)) AS earliest
            FROM messages
            WHERE source_type = 'news'
            """
        ).fetchone()
    value = row["earliest"] if row else None
    if not value:
        return None
    return datetime.fromisoformat(value)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="导入历史新闻标题CSV（id,title_cn,classification）到 messages 表，同时跑一遍候选召回"
    )
    parser.add_argument("--csv-path", type=Path, required=True)
    parser.add_argument("--database", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument(
        "--import-batch",
        default="fa_block_twitter_10000",
        help="记录在 raw_event 里的批次标识，方便以后追溯这批数据的来源",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()

    with args.csv_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        rows = list(reader)
    rows.sort(key=lambda r: int(r["id"]))

    anchor = _earliest_existing_news_time(args.database) or datetime.now(timezone.utc)
    # 这批数据没有真实时间戳，只有大致按时间递增的 id。
    # 用一个安全早于现有 news 消息的锚点时间，按 id 顺序给出合成的 received_at，
    # 保证之后按时间切分训练/验证/测试集时，这批数据整体排在更早的位置。
    base_time = anchor - timedelta(seconds=len(rows) + 10)

    mapper = default_mapper()

    inserted_count = 0
    skipped_count = 0
    with_candidate_count = 0
    empty_text_count = 0

    for offset, row in enumerate(rows):
        text = row["title_cn"].strip()
        if not text:
            empty_text_count += 1
            continue

        matches = mapper.identify(text)
        if matches:
            with_candidate_count += 1

        received_at = (base_time + timedelta(seconds=offset)).isoformat()
        record = {
            "source_id": row["id"],
            "source_type": "news",
            "source_url": "",
            "published_at": "",
            "text": text,
            "received_at": received_at,
            "status": "matched" if matches else "no_match",
            "predicted_codes": [match.canonical_code for match in matches],
            "companies": [match.to_dict() for match in matches],
            "upstream_candidates": [],
            "raw_event": {
                "import_batch": args.import_batch,
                "classification": row.get("classification", ""),
            },
        }
        _, inserted = save_message(record, model_version=MODEL_VERSION, path=args.database)
        if inserted:
            inserted_count += 1
        else:
            skipped_count += 1

    print(
        f"rows={len(rows)} inserted={inserted_count} skipped_dup={skipped_count} "
        f"empty_text={empty_text_count} with_candidate={with_candidate_count}"
    )


if __name__ == "__main__":
    main()

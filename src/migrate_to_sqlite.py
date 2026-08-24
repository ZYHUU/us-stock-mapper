from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from src.database import (
    DEFAULT_DB_PATH,
    database_counts,
    initialize,
    save_database_annotation,
    save_message,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def migrate(
    events_path: Path,
    annotations_path: Path,
    database_path: Path,
) -> dict[str, int]:
    initialize(database_path)
    imported_messages = 0
    imported_annotations = 0
    skipped_annotations = 0

    if events_path.exists():
        with events_path.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"消息文件第 {line_number} 行不是合法 JSON"
                    ) from error
                _, inserted = save_message(
                    record,
                    model_version="legacy-rules",
                    path=database_path,
                )
                imported_messages += int(inserted)

    if annotations_path.exists():
        with annotations_path.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                if not line.strip():
                    continue
                try:
                    annotation = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"标注文件第 {line_number} 行不是合法 JSON"
                    ) from error

                digest = hashlib.sha256(line.encode("utf-8")).hexdigest()
                source_ref = f"legacy:{line_number}:{digest}"
                try:
                    saved = save_database_annotation(
                        key=annotation["record_key"],
                        correct_codes=annotation.get("correct_codes", []),
                        scope_codes=annotation.get("scope_codes", []),
                        annotator=annotation.get("annotator", "legacy"),
                        confidence=annotation.get("confidence", "legacy"),
                        path=database_path,
                        source_ref=source_ref,
                        annotated_at=annotation.get("annotated_at"),
                        predicted_codes=annotation.get("predicted_codes", []),
                        decision=annotation.get("decision"),
                    )
                except KeyError:
                    skipped_annotations += 1
                    continue
                imported_annotations += int(saved is not None)

    return {
        "imported_messages": imported_messages,
        "imported_annotations": imported_annotations,
        "skipped_annotations": skipped_annotations,
        **database_counts(database_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="将现有 JSONL 数据迁移到 SQLite")
    parser.add_argument(
        "--events",
        type=Path,
        default=PROJECT_ROOT / "data" / "ws_events.jsonl",
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        default=PROJECT_ROOT / "data" / "annotations.jsonl",
    )
    parser.add_argument("--database", type=Path, default=DEFAULT_DB_PATH)
    args = parser.parse_args()
    result = migrate(args.events, args.annotations, args.database)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

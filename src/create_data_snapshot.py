from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from src.database import DEFAULT_DB_PATH, PROJECT_ROOT


DEFAULT_COMPANIES_PATH = PROJECT_ROOT / "data" / "companies.csv"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "snapshots"
DEFAULT_SNAPSHOT_NAME = "training-v1"
SNAPSHOT_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _readonly_connection(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def _database_summary(path: Path) -> dict[str, Any]:
    with closing(_readonly_connection(path)) as connection:
        integrity_check = str(connection.execute("PRAGMA quick_check").fetchone()[0])
        table_names = {
            str(row[0])
            for row in connection.execute(
                """
                SELECT name
                FROM sqlite_master
                WHERE type = 'table' AND name NOT LIKE 'sqlite_%'
                """
            )
        }
        required_tables = {"messages", "annotations"}
        missing_tables = sorted(required_tables - table_names)
        if missing_tables:
            raise ValueError(
                "源数据库缺少必要数据表：" + ", ".join(missing_tables)
            )

        table_counts = {
            table: int(
                connection.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
            )
            for table in sorted(table_names)
        }
        training_row = connection.execute(
            """
            WITH latest_annotations AS (
                SELECT a.*
                FROM annotations AS a
                WHERE a.id = (
                    SELECT MAX(newer.id)
                    FROM annotations AS newer
                    WHERE newer.message_id = a.message_id
                )
            )
            SELECT
                COUNT(*) AS valid_messages,
                SUM(CASE WHEN latest.id IS NOT NULL THEN 1 ELSE 0 END)
                    AS labeled_messages,
                SUM(CASE WHEN latest.id IS NULL THEN 1 ELSE 0 END)
                    AS unlabeled_messages,
                SUM(
                    CASE
                        WHEN latest.id IS NOT NULL
                         AND latest.correct_codes_json <> '[]'
                        THEN 1 ELSE 0
                    END
                ) AS positive_messages,
                MIN(m.received_at) AS first_received_at,
                MAX(m.received_at) AS last_received_at,
                MAX(latest.annotated_at) AS last_annotated_at
            FROM messages AS m
            LEFT JOIN latest_annotations AS latest
                ON latest.message_id = m.id
            WHERE TRIM(m.text) <> ''
            """
        ).fetchone()

    return {
        "integrity_check": integrity_check,
        "table_counts": table_counts,
        "training_data": {
            "valid_messages": int(training_row["valid_messages"] or 0),
            "labeled_messages": int(training_row["labeled_messages"] or 0),
            "unlabeled_messages": int(training_row["unlabeled_messages"] or 0),
            "positive_messages": int(training_row["positive_messages"] or 0),
            "first_received_at": training_row["first_received_at"],
            "last_received_at": training_row["last_received_at"],
            "last_annotated_at": training_row["last_annotated_at"],
        },
    }


def create_snapshot(
    source_database: Path = DEFAULT_DB_PATH,
    companies_source: Path = DEFAULT_COMPANIES_PATH,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    snapshot_name: str = DEFAULT_SNAPSHOT_NAME,
) -> dict[str, Any]:
    if not SNAPSHOT_NAME_PATTERN.fullmatch(snapshot_name):
        raise ValueError(
            "快照名称只能包含英文字母、数字、点、下划线和连字符"
        )

    source_database = source_database.resolve()
    companies_source = companies_source.resolve()
    output_root = output_root.resolve()
    snapshot_directory = output_root / snapshot_name

    if not source_database.is_file():
        raise FileNotFoundError(f"源数据库不存在：{source_database}")
    if not companies_source.is_file():
        raise FileNotFoundError(f"公司规则文件不存在：{companies_source}")
    if snapshot_directory.exists():
        raise FileExistsError(
            f"快照已存在，不会覆盖：{snapshot_directory}"
        )

    output_root.mkdir(parents=True, exist_ok=True)
    temporary_directory = output_root / f".{snapshot_name}.tmp-{uuid4().hex}"
    temporary_directory.mkdir()
    snapshot_database = temporary_directory / "stock_mapper.db"
    snapshot_companies = temporary_directory / "companies.csv"

    try:
        with closing(_readonly_connection(source_database)) as source_connection:
            with closing(sqlite3.connect(snapshot_database)) as target_connection:
                source_connection.backup(target_connection)

        cutoff_at = utc_now()
        shutil.copy2(companies_source, snapshot_companies)
        summary = _database_summary(snapshot_database)
        if summary["integrity_check"] != "ok":
            raise RuntimeError(
                "快照数据库完整性检查失败：" + summary["integrity_check"]
            )

        manifest = {
            "snapshot_name": snapshot_name,
            "cutoff_at": cutoff_at,
            "scope": "SQLite 在线备份完成前已提交的全部数据",
            "source_database": str(source_database),
            "files": {
                "database": {
                    "path": snapshot_database.name,
                    "sha256": sha256_file(snapshot_database),
                    "size_bytes": snapshot_database.stat().st_size,
                },
                "companies": {
                    "path": snapshot_companies.name,
                    "sha256": sha256_file(snapshot_companies),
                    "size_bytes": snapshot_companies.stat().st_size,
                },
            },
            **summary,
        }
        manifest_path = temporary_directory / "manifest.json"
        with manifest_path.open("w", encoding="utf-8", newline="\n") as file:
            json.dump(manifest, file, ensure_ascii=False, indent=2)
            file.write("\n")

        os.replace(temporary_directory, snapshot_directory)
    except Exception:
        shutil.rmtree(temporary_directory, ignore_errors=True)
        raise

    manifest["snapshot_directory"] = str(snapshot_directory)
    return manifest


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="固定 SQLite 消息、标注和公司规则的数据快照"
    )
    parser.add_argument("--name", default=DEFAULT_SNAPSHOT_NAME, help="快照名称")
    parser.add_argument(
        "--database",
        type=Path,
        default=DEFAULT_DB_PATH,
        help="持续采集使用的源 SQLite 数据库",
    )
    parser.add_argument(
        "--companies",
        type=Path,
        default=DEFAULT_COMPANIES_PATH,
        help="公司别名与人工规则 CSV",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="快照根目录",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    try:
        manifest = create_snapshot(
            source_database=args.database,
            companies_source=args.companies,
            output_root=args.output_root,
            snapshot_name=args.name,
        )
    except (FileExistsError, FileNotFoundError, RuntimeError, ValueError) as error:
        raise SystemExit(str(error)) from error

    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "stock_mapper.db"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def source_key(record: dict[str, Any]) -> str:
    source_id = str(record.get("source_id") or "").strip()
    if source_id:
        return f"id:{source_id}"

    raw_event = record.get("raw_event")
    if raw_event:
        normalized = json.dumps(raw_event, ensure_ascii=False, sort_keys=True)
        digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        return f"event:{digest}"

    text = str(record.get("text") or "").strip()
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return f"text:{digest}"


def connect(path: Path = DEFAULT_DB_PATH) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=30)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA journal_mode = WAL")
    connection.execute("PRAGMA busy_timeout = 30000")
    return connection


def initialize(path: Path = DEFAULT_DB_PATH) -> None:
    with closing(connect(path)) as connection, connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_key TEXT NOT NULL UNIQUE,
                source_id TEXT NOT NULL DEFAULT '',
                source_type TEXT NOT NULL DEFAULT 'unknown',
                source_url TEXT NOT NULL DEFAULT '',
                published_at TEXT NOT NULL DEFAULT '',
                text TEXT NOT NULL DEFAULT '',
                received_at TEXT NOT NULL,
                status TEXT NOT NULL,
                predicted_codes_json TEXT NOT NULL DEFAULT '[]',
                prediction_details_json TEXT NOT NULL DEFAULT '[]',
                upstream_candidates_json TEXT NOT NULL DEFAULT '[]',
                raw_event_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_messages_received_at
                ON messages(received_at);
            CREATE INDEX IF NOT EXISTS idx_messages_status
                ON messages(status);

            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                message_id INTEGER NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
                company_code TEXT NOT NULL,
                company_id TEXT NOT NULL DEFAULT '',
                company_name TEXT NOT NULL DEFAULT '',
                mention TEXT NOT NULL DEFAULT '',
                match_type TEXT NOT NULL DEFAULT '',
                confidence REAL NOT NULL DEFAULT 0,
                model_version TEXT NOT NULL,
                created_at TEXT NOT NULL,
                UNIQUE(message_id, company_code, model_version)
            );

            CREATE INDEX IF NOT EXISTS idx_predictions_company_code
                ON predictions(company_code);

            CREATE TABLE IF NOT EXISTS upstream_candidates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                message_id INTEGER NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
                stocker_id TEXT NOT NULL DEFAULT '',
                stocker_code TEXT NOT NULL DEFAULT '',
                stocker_name TEXT NOT NULL DEFAULT '',
                aliases_json TEXT NOT NULL DEFAULT '[]',
                candidate_type TEXT NOT NULL DEFAULT '',
                raw_candidate_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL,
                UNIQUE(message_id, stocker_id, stocker_code, stocker_name)
            );

            CREATE INDEX IF NOT EXISTS idx_upstream_candidates_code
                ON upstream_candidates(stocker_code);

            CREATE TABLE IF NOT EXISTS annotations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                message_id INTEGER NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
                annotated_at TEXT NOT NULL,
                decision TEXT NOT NULL,
                predicted_codes_json TEXT NOT NULL DEFAULT '[]',
                correct_codes_json TEXT NOT NULL DEFAULT '[]',
                scope_codes_json TEXT NOT NULL DEFAULT '[]',
                annotator TEXT NOT NULL DEFAULT 'human',
                confidence TEXT NOT NULL DEFAULT 'high',
                source_ref TEXT UNIQUE
            );

            CREATE INDEX IF NOT EXISTS idx_annotations_message_id
                ON annotations(message_id, id DESC);

            CREATE TABLE IF NOT EXISTS securities (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                canonical_code TEXT NOT NULL UNIQUE,
                asset_type TEXT NOT NULL,
                market TEXT NOT NULL DEFAULT '',
                exchange TEXT NOT NULL DEFAULT '',
                ticker TEXT NOT NULL DEFAULT '',
                name_en TEXT NOT NULL DEFAULT '',
                name_cn TEXT NOT NULL DEFAULT '',
                mapper_candidate INTEGER NOT NULL DEFAULT 0,
                source_url TEXT NOT NULL DEFAULT '',
                updated_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_securities_ticker
                ON securities(ticker);
            CREATE INDEX IF NOT EXISTS idx_securities_mapper_candidate
                ON securities(mapper_candidate, market);

            CREATE TABLE IF NOT EXISTS platform_instruments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                platform TEXT NOT NULL,
                contract_symbol TEXT NOT NULL,
                security_id INTEGER REFERENCES securities(id),
                base_asset TEXT NOT NULL DEFAULT '',
                quote_asset TEXT NOT NULL DEFAULT '',
                underlying_type TEXT NOT NULL DEFAULT '',
                underlying_subtypes_json TEXT NOT NULL DEFAULT '[]',
                status TEXT NOT NULL DEFAULT '',
                onboard_date TEXT NOT NULL DEFAULT '',
                active INTEGER NOT NULL DEFAULT 1,
                source_url TEXT NOT NULL DEFAULT '',
                raw_json TEXT NOT NULL DEFAULT '{}',
                synced_at TEXT NOT NULL,
                UNIQUE(platform, contract_symbol)
            );

            CREATE INDEX IF NOT EXISTS idx_platform_instruments_security_id
                ON platform_instruments(security_id);
            CREATE INDEX IF NOT EXISTS idx_platform_instruments_active
                ON platform_instruments(platform, active);

            CREATE TABLE IF NOT EXISTS shadow_predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                message_id INTEGER NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
                model_version TEXT NOT NULL,
                rule_codes_json TEXT NOT NULL DEFAULT '[]',
                model_codes_json TEXT NOT NULL DEFAULT '[]',
                candidates_json TEXT NOT NULL DEFAULT '[]',
                agrees INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                UNIQUE(message_id, model_version)
            );

            CREATE INDEX IF NOT EXISTS idx_shadow_predictions_agrees
                ON shadow_predictions(agrees);
            """
        )
        message_columns = {
            row["name"]
            for row in connection.execute("PRAGMA table_info(messages)").fetchall()
        }
        if "upstream_candidates_json" not in message_columns:
            connection.execute(
                """
                ALTER TABLE messages
                ADD COLUMN upstream_candidates_json TEXT NOT NULL DEFAULT '[]'
                """
            )
        if "source_type" not in message_columns:
            connection.execute(
                """
                ALTER TABLE messages
                ADD COLUMN source_type TEXT NOT NULL DEFAULT 'unknown'
                """
            )
        if "published_at" not in message_columns:
            connection.execute(
                """
                ALTER TABLE messages
                ADD COLUMN published_at TEXT NOT NULL DEFAULT ''
                """
            )
        connection.execute(
            """
            UPDATE messages
            SET source_type = 'twitter'
            WHERE source_type = 'unknown'
              AND (
                  source_url LIKE '%twitter.com/%'
                  OR source_url LIKE '%x.com/%'
              )
            """
        )


def save_message(
    record: dict[str, Any],
    model_version: str,
    path: Path = DEFAULT_DB_PATH,
) -> tuple[int, bool]:
    initialize(path)
    key = source_key(record)
    now = utc_now()
    companies = list(record.get("companies") or [])
    predicted_codes = list(record.get("predicted_codes") or [])
    upstream_candidates = list(record.get("upstream_candidates") or [])

    with closing(connect(path)) as connection, connection:
        cursor = connection.execute(
            """
            INSERT OR IGNORE INTO messages (
                source_key, source_id, source_type, source_url, published_at,
                text, received_at, status,
                predicted_codes_json, prediction_details_json,
                upstream_candidates_json, raw_event_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                key,
                str(record.get("source_id") or ""),
                str(record.get("source_type") or "unknown"),
                str(record.get("source_url") or ""),
                str(record.get("published_at") or ""),
                str(record.get("text") or ""),
                str(record.get("received_at") or now),
                str(record.get("status") or "no_match"),
                json.dumps(predicted_codes, ensure_ascii=False),
                json.dumps(companies, ensure_ascii=False),
                json.dumps(upstream_candidates, ensure_ascii=False),
                json.dumps(record.get("raw_event") or {}, ensure_ascii=False),
                now,
            ),
        )
        inserted = cursor.rowcount == 1
        message_row = connection.execute(
            "SELECT id FROM messages WHERE source_key = ?",
            (key,),
        ).fetchone()
        if message_row is None:
            raise RuntimeError("消息保存后无法读取")
        message_id = int(message_row["id"])

        for company in companies:
            connection.execute(
                """
                INSERT OR IGNORE INTO predictions (
                    message_id, company_code, company_id, company_name, mention,
                    match_type, confidence, model_version, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    message_id,
                    str(company.get("canonical_code") or ""),
                    str(company.get("company_id") or ""),
                    str(company.get("company_name") or ""),
                    str(company.get("mention") or ""),
                    str(company.get("match_type") or ""),
                    float(company.get("confidence") or 0),
                    model_version,
                    now,
                ),
            )

        for candidate in upstream_candidates:
            connection.execute(
                """
                INSERT OR IGNORE INTO upstream_candidates (
                    message_id, stocker_id, stocker_code, stocker_name,
                    aliases_json, candidate_type, raw_candidate_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    message_id,
                    str(candidate.get("stocker_id") or ""),
                    str(candidate.get("stocker_code") or ""),
                    str(candidate.get("stocker_name") or ""),
                    json.dumps(candidate.get("aliases") or [], ensure_ascii=False),
                    str(candidate.get("type") or ""),
                    json.dumps(candidate, ensure_ascii=False),
                    now,
                ),
            )

    return message_id, inserted


def annotation_summary(path: Path = DEFAULT_DB_PATH) -> dict[str, int]:
    initialize(path)
    with closing(connect(path)) as connection, connection:
        row = connection.execute(
            """
            SELECT
                COUNT(*) AS total,
                SUM(CASE WHEN latest.id IS NOT NULL THEN 1 ELSE 0 END) AS labeled
            FROM messages AS m
            LEFT JOIN annotations AS latest
                ON latest.id = (
                    SELECT a.id FROM annotations AS a
                    WHERE a.message_id = m.id
                    ORDER BY a.id DESC LIMIT 1
                )
            WHERE TRIM(m.text) <> ''
            """
        ).fetchone()
    total = int(row["total"] or 0)
    labeled = int(row["labeled"] or 0)
    return {"total": total, "labeled": labeled, "remaining": total - labeled}


def list_annotation_items(
    path: Path = DEFAULT_DB_PATH,
    limit: int = 100,
    offset: int = 0,
    unlabeled_only: bool = True,
) -> list[dict[str, Any]]:
    initialize(path)
    condition = "AND latest.id IS NULL" if unlabeled_only else ""
    query = f"""
        SELECT
            m.source_key, m.source_id, m.source_type, m.source_url,
            m.published_at, m.text,
            m.predicted_codes_json, m.upstream_candidates_json,
            latest.annotated_at, latest.decision,
            latest.predicted_codes_json AS annotation_predicted_codes_json,
            latest.correct_codes_json, latest.scope_codes_json,
            latest.annotator, latest.confidence
        FROM messages AS m
        LEFT JOIN annotations AS latest
            ON latest.id = (
                SELECT a.id FROM annotations AS a
                WHERE a.message_id = m.id
                ORDER BY a.id DESC LIMIT 1
            )
        WHERE TRIM(m.text) <> '' {condition}
        ORDER BY m.id
        LIMIT ? OFFSET ?
    """
    with closing(connect(path)) as connection, connection:
        rows = connection.execute(query, (limit, offset)).fetchall()

    items: list[dict[str, Any]] = []
    for row in rows:
        annotation = None
        if row["annotated_at"] is not None:
            annotation = {
                "record_key": row["source_key"],
                "annotated_at": row["annotated_at"],
                "decision": row["decision"],
                "predicted_codes": json.loads(
                    row["annotation_predicted_codes_json"]
                ),
                "correct_codes": json.loads(row["correct_codes_json"]),
                "scope_codes": json.loads(row["scope_codes_json"]),
                "annotator": row["annotator"],
                "confidence": row["confidence"],
            }
        items.append(
            {
                "record_key": row["source_key"],
                "source_id": row["source_id"],
                "source_type": row["source_type"],
                "source_url": row["source_url"],
                "published_at": row["published_at"],
                "text": row["text"],
                "predicted_codes": json.loads(row["predicted_codes_json"]),
                "upstream_candidates": json.loads(
                    row["upstream_candidates_json"]
                ),
                "annotation": annotation,
            }
        )
    return items


def save_database_annotation(
    key: str,
    correct_codes: list[str],
    scope_codes: list[str],
    annotator: str,
    confidence: str,
    path: Path = DEFAULT_DB_PATH,
    source_ref: str | None = None,
    annotated_at: str | None = None,
    predicted_codes: list[str] | None = None,
    decision: str | None = None,
) -> dict[str, Any] | None:
    initialize(path)
    unique_codes = list(dict.fromkeys(correct_codes))

    with closing(connect(path)) as connection, connection:
        message = connection.execute(
            """
            SELECT id, predicted_codes_json FROM messages
            WHERE source_key = ?
            """,
            (key,),
        ).fetchone()
        if message is None:
            raise KeyError(key)

        actual_predictions = (
            list(predicted_codes)
            if predicted_codes is not None
            else json.loads(message["predicted_codes_json"])
        )
        actual_decision = decision
        if actual_decision is None:
            if not unique_codes:
                actual_decision = "no_tracked_company"
            elif unique_codes == actual_predictions:
                actual_decision = "accepted"
            else:
                actual_decision = "corrected"

        annotation = {
            "record_key": key,
            "annotated_at": annotated_at or utc_now(),
            "decision": actual_decision,
            "predicted_codes": actual_predictions,
            "correct_codes": unique_codes,
            "scope_codes": sorted(scope_codes),
            "annotator": annotator,
            "confidence": confidence,
        }
        cursor = connection.execute(
            """
            INSERT OR IGNORE INTO annotations (
                message_id, annotated_at, decision, predicted_codes_json,
                correct_codes_json, scope_codes_json, annotator, confidence,
                source_ref
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                int(message["id"]),
                annotation["annotated_at"],
                annotation["decision"],
                json.dumps(actual_predictions, ensure_ascii=False),
                json.dumps(unique_codes, ensure_ascii=False),
                json.dumps(sorted(scope_codes), ensure_ascii=False),
                annotator,
                confidence,
                source_ref,
            ),
        )
        if cursor.rowcount == 0:
            return None
    return annotation


def save_shadow_prediction(
    message_id: int,
    model_version: str,
    rule_codes: list[str],
    model_codes: list[str],
    candidates: list[dict[str, Any]],
    path: Path = DEFAULT_DB_PATH,
) -> None:
    """记录影子运行结果：模型的判断只存起来，不影响 messages 里对外的正式输出。"""
    initialize(path)
    agrees = 1 if sorted(rule_codes) == sorted(model_codes) else 0
    with closing(connect(path)) as connection, connection:
        connection.execute(
            """
            INSERT OR IGNORE INTO shadow_predictions (
                message_id, model_version, rule_codes_json, model_codes_json,
                candidates_json, agrees, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                message_id,
                model_version,
                json.dumps(rule_codes, ensure_ascii=False),
                json.dumps(model_codes, ensure_ascii=False),
                json.dumps(candidates, ensure_ascii=False),
                agrees,
                utc_now(),
            ),
        )


def shadow_run_summary(
    model_version: str, path: Path = DEFAULT_DB_PATH
) -> dict[str, int]:
    initialize(path)
    with closing(connect(path)) as connection, connection:
        row = connection.execute(
            """
            SELECT
                COUNT(*) AS total,
                SUM(CASE WHEN agrees = 1 THEN 1 ELSE 0 END) AS agrees
            FROM shadow_predictions
            WHERE model_version = ?
            """,
            (model_version,),
        ).fetchone()
    total = int(row["total"] or 0)
    agrees = int(row["agrees"] or 0)
    return {"total": total, "agrees": agrees, "disagrees": total - agrees}


def database_counts(path: Path = DEFAULT_DB_PATH) -> dict[str, int]:
    initialize(path)
    with closing(connect(path)) as connection, connection:
        return {
            "messages": int(
                connection.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
            ),
            "predictions": int(
                connection.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
            ),
            "upstream_candidates": int(
                connection.execute(
                    "SELECT COUNT(*) FROM upstream_candidates"
                ).fetchone()[0]
            ),
            "annotations": int(
                connection.execute("SELECT COUNT(*) FROM annotations").fetchone()[0]
            ),
        }

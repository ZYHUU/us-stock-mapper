from __future__ import annotations

import json
from contextlib import closing
from pathlib import Path
from typing import Any

from src.database import DEFAULT_DB_PATH, connect, initialize


def _eligible_query(model_count: int) -> str:
    placeholders = ", ".join("?" for _ in range(model_count))
    return f"""
        WITH eligible AS (
            SELECT
                message_id,
                MAX(created_at) AS last_shadow_at,
                CASE
                    WHEN COUNT(DISTINCT model_codes_json) = 1 THEN 0
                    ELSE 1
                END AS priority_group
            FROM shadow_predictions
            WHERE model_version IN ({placeholders})
            GROUP BY message_id
            HAVING COUNT(DISTINCT model_version) = ?
               AND MIN(agrees) = 0
        ),
        latest_annotations AS (
            SELECT a.*
            FROM annotations AS a
            WHERE a.id = (
                SELECT MAX(newer.id)
                FROM annotations AS newer
                WHERE newer.message_id = a.message_id
            )
        )
    """


def shadow_review_summary(
    model_versions: list[str],
    path: Path = DEFAULT_DB_PATH,
) -> dict[str, int]:
    """汇总指定模型均已打分的影子分歧和人工复核进度。"""
    versions = list(dict.fromkeys(model_versions))
    if not versions:
        return {"total": 0, "reviewed": 0, "pending": 0}

    initialize(path)
    query = _eligible_query(len(versions)) + """
        SELECT
            COUNT(*) AS total,
            SUM(CASE WHEN latest.id IS NOT NULL THEN 1 ELSE 0 END) AS reviewed
        FROM eligible
        LEFT JOIN latest_annotations AS latest
            ON latest.message_id = eligible.message_id
    """
    with closing(connect(path)) as connection, connection:
        row = connection.execute(query, (*versions, len(versions))).fetchone()

    total = int(row["total"] or 0)
    reviewed = int(row["reviewed"] or 0)
    return {"total": total, "reviewed": reviewed, "pending": total - reviewed}


def list_shadow_review_items(
    model_versions: list[str],
    path: Path = DEFAULT_DB_PATH,
    limit: int = 100,
    offset: int = 0,
    unreviewed_only: bool = True,
) -> list[dict[str, Any]]:
    """返回规则与任一影子模型存在分歧的消息。"""
    versions = list(dict.fromkeys(model_versions))
    if not versions:
        return []

    initialize(path)
    annotation_condition = "AND latest.id IS NULL" if unreviewed_only else ""
    shadow_placeholders = ", ".join("?" for _ in versions)
    query = _eligible_query(len(versions)) + f"""
        , selected_messages AS (
            SELECT
                eligible.message_id,
                eligible.last_shadow_at,
                eligible.priority_group
            FROM eligible
            LEFT JOIN latest_annotations AS latest
                ON latest.message_id = eligible.message_id
            WHERE 1 = 1 {annotation_condition}
            ORDER BY eligible.priority_group, eligible.last_shadow_at DESC
            LIMIT ? OFFSET ?
        )
        SELECT
            selected.priority_group, selected.last_shadow_at,
            m.id AS message_id, m.source_key, m.source_id, m.source_type,
            m.source_url, m.published_at, m.text, m.predicted_codes_json,
            m.upstream_candidates_json,
            sp.model_version, sp.model_codes_json, sp.candidates_json,
            sp.agrees, sp.created_at,
            latest.annotated_at, latest.correct_codes_json,
            latest.annotator, latest.confidence
        FROM selected_messages AS selected
        JOIN messages AS m ON m.id = selected.message_id
        JOIN shadow_predictions AS sp
            ON sp.message_id = selected.message_id
           AND sp.model_version IN ({shadow_placeholders})
        LEFT JOIN latest_annotations AS latest
            ON latest.message_id = selected.message_id
        ORDER BY selected.priority_group, selected.last_shadow_at DESC, sp.model_version
    """
    parameters = (*versions, len(versions), limit, offset, *versions)
    with closing(connect(path)) as connection, connection:
        rows = connection.execute(query, parameters).fetchall()

    grouped: dict[int, dict[str, Any]] = {}
    for row in rows:
        message_id = int(row["message_id"])
        if message_id not in grouped:
            annotation = None
            if row["annotated_at"] is not None:
                annotation = {
                    "annotated_at": row["annotated_at"],
                    "correct_codes": json.loads(row["correct_codes_json"]),
                    "annotator": row["annotator"],
                    "confidence": row["confidence"],
                }
            grouped[message_id] = {
                "message_id": message_id,
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
                "shadow_predictions": [],
                "review_reasons": [],
                "priority": 100 if int(row["priority_group"]) == 0 else 90,
                "last_shadow_at": row["last_shadow_at"],
                "annotation": annotation,
            }
        grouped[message_id]["shadow_predictions"].append(
            {
                "model_version": row["model_version"],
                "model_codes": json.loads(row["model_codes_json"]),
                "candidates": json.loads(row["candidates_json"]),
                "agrees_with_rule": bool(row["agrees"]),
                "created_at": row["created_at"],
            }
        )

    items = list(grouped.values())
    for item in items:
        model_code_sets = {
            tuple(sorted(prediction["model_codes"]))
            for prediction in item["shadow_predictions"]
        }
        if len(model_code_sets) == 1:
            item["review_reasons"].append("影子模型一致但与规则不同")
        else:
            item["review_reasons"].append("影子模型之间存在分歧")

        disagreeing_versions = [
            prediction["model_version"]
            for prediction in item["shadow_predictions"]
            if not prediction["agrees_with_rule"]
        ]
        if disagreeing_versions:
            item["review_reasons"].append(
                "规则分歧：" + ", ".join(disagreeing_versions)
            )

        candidate_codes = {
            str(candidate.get("canonical_code") or "")
            for prediction in item["shadow_predictions"]
            for candidate in prediction["candidates"]
            if str(candidate.get("canonical_code") or "")
        }
        if len(candidate_codes) > 1:
            item["priority"] += 5
            item["review_reasons"].append("多候选消息")

    return sorted(
        items,
        key=lambda item: (item["priority"], item["last_shadow_at"]),
        reverse=True,
    )

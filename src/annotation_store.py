from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def record_key(record: dict[str, Any]) -> str:
    source_id = str(record.get("source_id") or "").strip()
    if source_id:
        return f"id:{source_id}"

    text = str(record.get("text") or "").strip()
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return f"text:{digest}"


def load_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []

    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"采集文件第 {line_number} 行不是合法 JSON") from error
            if str(record.get("text") or "").strip():
                records.append(record)
    return records


def load_annotations(path: Path) -> dict[str, dict[str, Any]]:
    annotations: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return annotations

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                annotation = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"标注文件第 {line_number} 行不是合法 JSON") from error
            annotations[annotation["record_key"]] = annotation
    return annotations


def save_annotation(
    path: Path,
    key: str,
    correct_codes: list[str],
    predicted_codes: list[str],
    scope_codes: list[str] | None = None,
    annotator: str = "human",
    confidence: str = "high",
) -> dict[str, Any]:
    unique_codes = list(dict.fromkeys(correct_codes))
    if not unique_codes:
        decision = "no_tracked_company"
    elif unique_codes == predicted_codes:
        decision = "accepted"
    else:
        decision = "corrected"

    annotation = {
        "record_key": key,
        "annotated_at": datetime.now(timezone.utc).isoformat(),
        "decision": decision,
        "predicted_codes": predicted_codes,
        "correct_codes": unique_codes,
        "scope_codes": sorted(scope_codes or []),
        "annotator": annotator,
        "confidence": confidence,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as file:
        file.write(json.dumps(annotation, ensure_ascii=False) + "\n")
    return annotation

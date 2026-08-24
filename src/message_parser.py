from __future__ import annotations

import json
from typing import Any

from src.mapper import USStockMapper


# 优先使用中文翻译字段；字段为空时，继续使用原文字段。
TEXT_FIELDS = (
    "content_cn",
    "content",
    "title_cn",
    "title",
    "new_full_text_cn",
    "new_full_text",
    "quo_text_cn",
    "quo_text",
    "full_text",
    "text",
)

EMPTY_VALUES = {"", "null", "{}", "[]"}


def extract_upstream_candidates(event: dict[str, Any]) -> list[dict[str, Any]]:
    """读取上游正则召回结果。它们是候选，不等同于最终语义判断。"""
    value = event.get("stoks")
    if value is None:
        value = event.get("stocks")
    if value is None and event.get("stocker_code"):
        value = event

    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return []
    if isinstance(value, dict):
        value = [value]
    if not isinstance(value, list):
        return []

    candidates: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for item in value:
        if not isinstance(item, dict):
            continue
        aliases = item.get("aliases") or []
        if isinstance(aliases, str):
            aliases = [aliases]
        aliases = [str(alias).strip() for alias in aliases if str(alias).strip()]

        candidate = {
            "stocker_id": str(item.get("stocker_id") or ""),
            "stocker_code": str(item.get("stocker_code") or "").strip().upper(),
            "stocker_name": str(item.get("stocker_name") or "").strip(),
            "aliases": aliases,
            "type": str(item.get("type") or ""),
        }
        key = (
            candidate["stocker_id"],
            candidate["stocker_code"],
            candidate["stocker_name"],
        )
        if key in seen or not any(key):
            continue
        seen.add(key)
        candidates.append(candidate)
    return candidates


def detect_source_type(event: dict[str, Any]) -> str:
    explicit_type = str(event.get("source_type") or "").strip().lower()
    if explicit_type:
        return explicit_type

    url = str(event.get("item_url") or event.get("content_url") or "").lower()
    if "twitter.com/" in url or "x.com/" in url or event.get("id_str"):
        return "twitter"
    if event.get("content_url") or event.get("types"):
        return "news"
    return "unknown"


def extract_text(event: dict[str, Any]) -> str:
    """从后端推送事件中提取可用于公司识别的文本，并删除重复内容。"""
    texts: list[str] = []
    seen: set[str] = set()

    for field in TEXT_FIELDS:
        value = event.get(field)
        if not isinstance(value, str):
            continue

        text = value.strip()
        if text.casefold() in EMPTY_VALUES or text in seen:
            continue

        seen.add(text)
        texts.append(text)

    return "\n".join(texts)


def identify_event(event: dict[str, Any], mapper: USStockMapper) -> dict[str, object]:
    source_type = detect_source_type(event)
    text = extract_text(event)
    matches = mapper.identify(text)
    upstream_candidates = extract_upstream_candidates(event)
    return {
        "source_id": str(
            event.get("id_str") or event.get("id") or event.get("new_id") or ""
        ),
        "source_type": source_type,
        "source_url": str(event.get("item_url") or event.get("content_url") or ""),
        "published_at": str(event.get("create_time") or ""),
        "text": text,
        "status": "matched" if matches else "no_match",
        "companies": [match.to_dict() for match in matches],
        "matches": matches,
        "upstream_candidates": upstream_candidates,
    }

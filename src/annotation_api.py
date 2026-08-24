from __future__ import annotations

from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from src.database import (
    DEFAULT_DB_PATH,
    annotation_summary,
    list_annotation_items,
    save_database_annotation,
)
from src.mapper import default_mapper


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PAGE_PATH = PROJECT_ROOT / "static" / "annotate.html"

app = FastAPI(title="美股消息标注工具", version="0.1.0")


class AnnotationRequest(BaseModel):
    record_key: str
    correct_codes: list[str]
    annotator: str = "human"
    confidence: Literal["high", "medium", "low"] = "high"


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return PAGE_PATH.read_text(encoding="utf-8")


@app.get("/api/items")
def items(
    limit: int = 100,
    offset: int = 0,
    unlabeled_only: bool = True,
) -> dict[str, object]:
    limit = max(1, min(limit, 500))
    offset = max(0, offset)
    mapper = default_mapper()
    compact_items = list_annotation_items(
        DEFAULT_DB_PATH,
        limit=limit,
        offset=offset,
        unlabeled_only=unlabeled_only,
    )

    companies = [
        {
            "company_name": company.company_name,
            "canonical_code": company.canonical_code,
        }
        for company in mapper.companies
    ]
    return {
        "items": compact_items,
        "companies": companies,
        "summary": annotation_summary(DEFAULT_DB_PATH),
    }


@app.post("/api/annotations")
def annotate(request: AnnotationRequest) -> dict[str, object]:
    mapper = default_mapper()
    valid_codes = {company.canonical_code for company in mapper.companies}
    unknown_codes = set(request.correct_codes) - valid_codes
    if unknown_codes:
        raise HTTPException(
            status_code=400,
            detail=f"公司库中不存在：{', '.join(sorted(unknown_codes))}",
        )

    try:
        annotation = save_database_annotation(
            key=request.record_key,
            correct_codes=request.correct_codes,
            scope_codes=sorted(valid_codes),
            annotator=request.annotator,
            confidence=request.confidence,
            path=DEFAULT_DB_PATH,
        )
    except KeyError as error:
        raise HTTPException(status_code=404, detail="找不到这条消息") from error
    return {"ok": True, "annotation": annotation}

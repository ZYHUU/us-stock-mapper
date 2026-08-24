from __future__ import annotations

from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from src.database import DEFAULT_DB_PATH, save_database_annotation
from src.mapper import default_mapper
from src.semantic_matcher import LIGHTGBM_MODEL_VERSION, LR_MODEL_VERSION
from src.shadow_review import list_shadow_review_items, shadow_review_summary


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PAGE_PATH = PROJECT_ROOT / "static" / "shadow_review.html"
MODEL_VERSIONS = [LR_MODEL_VERSION, LIGHTGBM_MODEL_VERSION]

app = FastAPI(title="影子模型分歧复核", version="0.1.0")


class ReviewAnnotationRequest(BaseModel):
    record_key: str
    correct_codes: list[str]
    confidence: Literal["high", "medium", "low"] = "high"


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return PAGE_PATH.read_text(encoding="utf-8")


@app.get("/api/review-items")
def review_items(
    limit: int = 100,
    offset: int = 0,
    unreviewed_only: bool = True,
) -> dict[str, object]:
    limit = max(1, min(limit, 500))
    offset = max(0, offset)
    mapper = default_mapper()
    companies = [
        {
            "company_name": company.company_name,
            "canonical_code": company.canonical_code,
        }
        for company in mapper.companies
    ]
    return {
        "items": list_shadow_review_items(
            MODEL_VERSIONS,
            DEFAULT_DB_PATH,
            limit=limit,
            offset=offset,
            unreviewed_only=unreviewed_only,
        ),
        "companies": companies,
        "model_versions": MODEL_VERSIONS,
        "summary": shadow_review_summary(MODEL_VERSIONS, DEFAULT_DB_PATH),
    }


@app.post("/api/review-annotations")
def review_annotation(request: ReviewAnnotationRequest) -> dict[str, object]:
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
            annotator="shadow_review",
            confidence=request.confidence,
            path=DEFAULT_DB_PATH,
        )
    except KeyError as error:
        raise HTTPException(status_code=404, detail="找不到这条消息") from error
    return {"ok": True, "annotation": annotation}

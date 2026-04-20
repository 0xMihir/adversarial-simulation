"""
GET /api/annotations/{id}   — return full CaseAnnotation or 404
PUT /api/annotations/{id}   — partial update with corrections log
"""
from fastapi import APIRouter, HTTPException

from ..models.annotation import CaseAnnotation, CaseAnnotationUpdate
from ..services import annotations as ann_service

router = APIRouter()


@router.get("/{case_id}", response_model=CaseAnnotation)
def get_annotation(case_id: str):
    ann = ann_service.read_annotation(case_id)
    if ann is None:
        raise HTTPException(status_code=404, detail=f"Annotation not found: {case_id}")
    return ann


@router.put("/{case_id}", response_model=CaseAnnotation)
def update_annotation(case_id: str, update: CaseAnnotationUpdate):
    try:
        return ann_service.apply_update(case_id, update)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

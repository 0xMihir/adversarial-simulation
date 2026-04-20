"""
GET /api/cases                  — list all .far files with annotation status
POST /api/cases/{id}/process    — run full pipeline, write annotation JSON
GET  /api/cases/{id}/scene      — return ParsedScene only (lightweight)
"""

from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, HTTPException

from ..models.annotation import CaseAnnotation, CaseSummary, ElementStatus
from ..services import annotations as ann_service
from ..services.centerline_extractor import extract_centerlines
from ..services.clf_service import flush, get_cache, get_pipeline
from ..services.faro_parser import parse_scene
from ..services.texture_service import get_cache as get_texture_cache
from ..services.trajectory_fitter import fit_trajectories


router = APIRouter()

# Path to NHTSA CISS .far files — resolved relative to this file
DATA_DIR = Path(__file__).parents[3] / "data" / "nhtsa-ciss" / "data" / "output"


def _find_far_file(case_id: str) -> Path:
    """Find the .far file for a given case_id."""
    # Try exact filename match first
    direct = DATA_DIR / f"{case_id}.far"
    if direct.exists():
        return direct
    # Search recursively
    matches = list(DATA_DIR.rglob(f"*{case_id}*.far"))
    if matches:
        return matches[0]
    raise HTTPException(
        status_code=404, detail=f"No .far file found for case: {case_id}"
    )


def _case_id_from_path(path: Path) -> str:
    return path.stem


@router.get("", response_model=list[CaseSummary])
def list_cases():
    """List all available .far files with their annotation status."""
    if not DATA_DIR.exists():
        return []

    far_files = sorted(DATA_DIR.rglob("*.far"))
    summaries: list[CaseSummary] = []

    for far_path in far_files:
        case_id = _case_id_from_path(far_path)
        ann = ann_service.read_annotation(case_id)

        summaries.append(
            CaseSummary(
                id=case_id,
                filename=far_path.name,
                annotated=ann is not None,
                workflow_status=ann.workflow_status if ann else "not_started",
                updated_at=ann.updated_at if ann else None,
                lane_count=len(ann.lanes) if ann else 0,
                vehicle_count=len(ann.vehicles) if ann else 0,
                auto_confidence=ann.auto_confidence if ann else None,
            )
        )

    return summaries


@router.post("/{case_id}/process", response_model=CaseAnnotation)
def process_case(case_id: str):
    """
    Run full preprocessing pipeline on a .far file.
    Writes annotations/{case_id}.json.
    Idempotent: warns if human edits already exist (does not overwrite them).
    """
    far_path = _find_far_file(case_id)
    existing = ann_service.read_annotation(case_id)

    # Check if human edits exist
    has_human_edits = False
    if existing:
        has_human_edits = any(
            (
                lane.status
                in (
                    ElementStatus.CONFIRMED,
                    ElementStatus.CORRECTED,
                    ElementStatus.REJECTED,
                )
            )
            for lane in existing.lanes
        ) or any(
            (
                v.status
                in (
                    ElementStatus.CONFIRMED,
                    ElementStatus.CORRECTED,
                    ElementStatus.REJECTED,
                )
            )
            for v in existing.vehicles
        )

    # Always re-run preprocessing to regenerate auto data
    scene = parse_scene(str(far_path), case_id, get_pipeline(), get_cache(), get_texture_cache())
    auto_lanes = extract_centerlines(scene)
    auto_vehicles = fit_trajectories(scene)

    # Compute aggregate confidence
    lane_conf = len(auto_lanes) / max(len(auto_lanes), 1)
    veh_conf = sum(v.classification_score for v in scene.vehicles) / max(
        len(scene.vehicles), 1
    )
    auto_confidence = round((lane_conf + veh_conf) / 2, 3)

    now = datetime.now(timezone.utc).isoformat()

    if has_human_edits and existing:
        # Preserve human-edited lanes/vehicles; only update auto_centerlines + scene
        annotation = CaseAnnotation(
            case_id=case_id,
            far_filename=far_path.name,
            scene=scene,
            auto_centerlines=auto_lanes,
            lanes=existing.lanes,
            vehicles=existing.vehicles,
            lane_connections=existing.lane_connections,
            corrections=existing.corrections,
            created_at=existing.created_at,
            updated_at=now,
            annotator=existing.annotator,
            workflow_status=existing.workflow_status,
            auto_confidence=auto_confidence,
        )
    else:
        annotation = CaseAnnotation(
            case_id=case_id,
            far_filename=far_path.name,
            scene=scene,
            auto_centerlines=auto_lanes,
            lanes=list(auto_lanes),  # working copy starts as auto
            vehicles=auto_vehicles,
            lane_connections=[],
            corrections=[],
            created_at=existing.created_at if existing else now,
            updated_at=now,
            workflow_status="not_started",
            auto_confidence=auto_confidence,
        )

    ann_service.write_annotation(annotation)
    flush()
    return annotation


@router.get("/{case_id}/scene")
def get_scene(case_id: str):
    """Return ParsedScene only — lightweight for previews."""
    ann = ann_service.read_annotation(case_id)
    if ann is None:
        raise HTTPException(
            status_code=404, detail=f"Case not yet processed: {case_id}"
        )
    return ann.scene

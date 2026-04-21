"""
Atomic JSON read/write for per-case annotation files.
Storage: annotation/annotations/{case_id}.json
"""
import json
import os
import tempfile
from pathlib import Path

from schema.annotation import CaseAnnotation, CaseAnnotationUpdate

ANNOTATIONS_DIR = Path(__file__).parents[2] / "annotations"
ANNOTATIONS_DIR.mkdir(exist_ok=True)


def _annotation_path(case_id: str) -> Path:
    return ANNOTATIONS_DIR / f"{case_id}.json"


def read_annotation(case_id: str) -> CaseAnnotation | None:
    path = _annotation_path(case_id)
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    return CaseAnnotation.model_validate(data)


def write_annotation(annotation: CaseAnnotation) -> None:
    """Atomic write: write to temp file then rename."""
    path = _annotation_path(annotation.case_id)
    data = annotation.model_dump_json(indent=2)
    # Write to temp in the same directory for atomic rename
    fd, tmp_path = tempfile.mkstemp(dir=ANNOTATIONS_DIR, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(data)
        os.replace(tmp_path, path)
    except Exception:
        os.unlink(tmp_path)
        raise


def apply_update(case_id: str, update: CaseAnnotationUpdate) -> CaseAnnotation:
    """Apply a partial update to an existing annotation and save."""
    from datetime import datetime, timezone

    ann = read_annotation(case_id)
    if ann is None:
        raise FileNotFoundError(f"Annotation not found: {case_id}")

    if update.lanes is not None:
        ann.lanes = update.lanes
    if update.vehicles is not None:
        ann.vehicles = update.vehicles
    if update.lane_connections is not None:
        ann.lane_connections = update.lane_connections
    if update.workflow_status is not None:
        ann.workflow_status = update.workflow_status
    if update.annotator is not None:
        ann.annotator = update.annotator
    if update.corrections is not None:
        ann.corrections.extend(update.corrections)
    if update.hidden_roadway_ids is not None:
        ann.hidden_roadway_ids = update.hidden_roadway_ids

    ann.updated_at = datetime.now(timezone.utc).isoformat()
    write_annotation(ann)
    return ann


def list_annotated_ids() -> list[str]:
    return [p.stem for p in ANNOTATIONS_DIR.glob("*.json")]

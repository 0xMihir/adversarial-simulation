"""
Wraps preprocessing/faro.py and maps output to ParsedScene Pydantic model.
Captures raw control points alongside resampled polylines.
"""
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from faro import FaroSceneGraphReader
from schema.scene import (
    AffineMatrix,
    ImageElement,
    ParsedScene,
    Point2D,
    ScaleBar,
    SceneElement,
    TextElement,
    VehicleDetection,
)


def _pts_to_point2d(pts: list) -> list[Point2D]:
    return [Point2D(x=float(p[0]), y=float(p[1])) for p in pts]


def _matrix_to_model(mat: np.ndarray) -> AffineMatrix:
    return AffineMatrix(values=mat.tolist())


def _extract_obb(symbol: dict) -> list[Point2D]:
    """
    Extract an oriented bounding box from a vehicle symbol.
    Uses the symbol's transformed bounding box corners.
    """
    bbox = symbol.get("bbox", (0, 0, 0, 0))
    xmin, ymin, xmax, ymax = bbox
    center = symbol.get("transformed_center", ((xmin + xmax) / 2, (ymin + ymax) / 2))
    mat = symbol.get("transform", np.eye(3))

    # Generate 4 corners from local bbox, apply transform
    corners_local = [
        (xmin, ymin),
        (xmax, ymin),
        (xmax, ymax),
        (xmin, ymax),
    ]
    transformed = FaroSceneGraphReader.__new__(FaroSceneGraphReader)
    transformed._apply_transform = lambda pts, m: _apply_transform_static(pts, m)

    corners_world = _apply_transform_static(corners_local, mat)
    return _pts_to_point2d(corners_world)


def _apply_transform_static(points: list, matrix: np.ndarray) -> list:
    if not points:
        return []
    pts = np.array(points)
    ones = np.ones((pts.shape[0], 1))
    pts_homo = np.hstack([pts, ones])
    transformed = (matrix @ pts_homo.T).T
    return [tuple(row[:2]) for row in transformed]


def _heading_from_obb(symbol: dict) -> float:
    """Estimate vehicle heading from the major axis of its OBB."""
    mat = symbol.get("transform", np.eye(3))
    # Extract rotation angle from the 2x2 upper-left of the transform
    angle = float(np.arctan2(mat[1, 0], mat[0, 0]))
    return angle


def _confidence_from_symbol(symbol: dict) -> float:
    prob = symbol.get("predicted_probability")
    if prob is not None:
        return float(prob)
    if symbol.get("vehicle2d"):
        return 1.0
    return 0.5


def parse_scene(far_path: str, case_id: str, clf_pipeline=None, cls_cache=None, texture_cache=None) -> ParsedScene:
    """
    Parse a FARO .far file and return a comprehensive ParsedScene.

    Args:
        far_path: Absolute path to the .far file.
        case_id: Identifier for this crash case.
        clf_pipeline: Optional HuggingFace zero-shot classification pipeline.
                      If None, vehicle detection uses heuristics only.
        cls_cache: Optional dict cache for classification results.
        texture_cache: Optional flat {normalized_key: base64_data} dict for sign textures.
    """
    if cls_cache is None:
        cls_cache = {}

    reader = FaroSceneGraphReader(far_path, clf_pipeline, cls_cache, texture_cache)
    scene_objects = reader.parse()

    elements: list[SceneElement] = []
    element_id_map: dict[int, str] = {}  # id(raw_item) -> element.id

    roadway_indices: list[int] = []
    road_marking_indices: list[int] = []
    other_indices: list[int] = []

    # --- Flatten all primitives into SceneElement list ---
    def _add_element(prim: dict, idx: int) -> str:
        eid = str(uuid.uuid4())[:8]
        _tcp = prim.get("transformed_control_points")
        _tv = prim.get("transformed_verts")
        cp = _tcp if _tcp is not None and len(_tcp) > 0 else (_tv if _tv is not None and len(_tv) > 0 else [])
        _bh = prim.get("transformed_bezier_handles")
        bh = _bh if _bh is not None and len(_bh) > 0 else []
        rv = _tv if _tv is not None and len(_tv) > 0 else []

        elem = SceneElement(
            id=eid,
            faro_item_name=prim.get("name"),
            layer=prim.get("layer"),
            element_type=prim.get("type", "unknown"),
            control_points=_pts_to_point2d(cp),
            bezier_handles=_pts_to_point2d(bh),
            interpolation_method=prim.get("interpolation_method", "passthrough"),
            resampled_points=_pts_to_point2d(rv),
            transform=_matrix_to_model(prim.get("transform", np.eye(3))),
            is_closed=prim.get("closed", False),
            is_dashed=prim.get("dashed", False),
            line_width=1.5 if prim.get("thick") else None,
            bbox=tuple(prim.get("bbox", (0.0, 0.0, 0.0, 0.0))),
        )
        elements.append(elem)
        element_id_map[id(prim)] = eid
        return eid

    for prim in scene_objects.get("roadway", []):
        idx = len(elements)
        _add_element(prim, idx)
        roadway_indices.append(idx)

    for prim in scene_objects.get("road_markings", []):
        # Road markings are symbols; flatten their sub-items
        if prim.get("type") == "symbol":
            for sub in prim.get("items", []):
                if sub.get("type") != "label":
                    idx = len(elements)
                    _add_element(sub, idx)
                    road_marking_indices.append(idx)
        else:
            idx = len(elements)
            _add_element(prim, idx)
            road_marking_indices.append(idx)

    for prim in scene_objects.get("misc", []):
        idx = len(elements)
        if prim.get("type") == "symbol":
            for sub in prim.get("items", []):
                if sub.get("type") != "label":
                    idx2 = len(elements)
                    _add_element(sub, idx2)
                    other_indices.append(idx2)
        elif prim.get("verts"):
            _add_element(prim, idx)
            other_indices.append(idx)

    # --- Image elements (road signs) ---
    images: list[ImageElement] = []
    for i, prim in enumerate(scene_objects.get("images", [])):
        tc = prim.get("transformed_center", (0.0, 0.0))
        images.append(ImageElement(
            id=f"img_{i}",
            center=Point2D(x=float(tc[0]), y=float(tc[1])),
            sizx=float(prim.get("sizx", 0.0)),
            sizy=float(prim.get("sizy", 0.0)),
            oriz=float(prim.get("oriz", 0.0)),
            img=prim.get("img", ""),
            layer=prim.get("layer"),
        ))

    # --- Text elements ---
    texts: list[TextElement] = []
    for i, prim in enumerate(scene_objects.get("texts", [])):
        tc = prim.get("transformed_center", (0, 0))
        texts.append(TextElement(
            id=f"txt_{i}",
            text=prim.get("text") or "",
            position=Point2D(x=float(tc[0]), y=float(tc[1])),
            rotation=prim.get("oriz", 0.0),
        ))

    # --- Vehicle detections ---
    vehicles: list[VehicleDetection] = []
    for i, sym in enumerate(scene_objects.get("vehicles", [])):
        vid = f"V{i + 1}"
        obb_pts = _extract_obb(sym)
        tc = sym.get("transformed_center", (0, 0))
        associated_text = sym.get("associated_text", [])
        label_text = associated_text[0] if associated_text else None

        # Collect sub-element IDs
        src_ids: list[str] = []
        for sub in sym.get("items", []):
            if id(sub) in element_id_map:
                src_ids.append(element_id_map[id(sub)])

        vehicles.append(VehicleDetection(
            id=vid,
            source_element_ids=src_ids,
            obb=obb_pts if len(obb_pts) == 4 else [],
            center=Point2D(x=float(tc[0]), y=float(tc[1])),
            heading=_heading_from_obb(sym),
            classification_score=_confidence_from_symbol(sym),
            predicted_class=sym.get("predicted_class"),
            label_text=label_text,
        ))

    # --- Scale bar ---
    scale_bar: ScaleBar | None = None
    raw_sb = scene_objects.get("scalebar")
    if raw_sb:
        sx, sy = raw_sb.get("size", (0, 0))
        scale_bar = ScaleBar(
            length_pixels=float(max(sx, sy)),
            length_real=1.0,  # FARO scalebar encodes real length in its display text
            unit="ft",
        )

    return ParsedScene(
        case_id=case_id,
        far_filename=str(Path(far_path).name),
        parsed_at=datetime.now(timezone.utc).isoformat(),
        coordinate_unit="ft",
        scale_bar=scale_bar,
        elements=elements,
        texts=texts,
        images=images,
        vehicles=vehicles,
        roadway_indices=roadway_indices,
        road_marking_indices=road_marking_indices,
        other_indices=other_indices,
    )

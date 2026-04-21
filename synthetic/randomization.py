"""
Nine named randomization modules (§1.3).

Each function is a pure function: deterministic given the rng, mutates no shared state,
returns new objects rather than modifying inputs.

Module naming (referenced by §1.3 number in docstrings):
    drop_primitives       — Module 1
    resample_vertices     — Module 2
    jitter_vertices       — Module 3
    vary_crosswalk        — Module 4
    add_annotation_clutter — Module 5
    subset_network        — Module 6
    transform_layout      — Module 7
    add_vehicle_markings  — Module 8
    apply_faro_styling    — Module 9
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
from scipy.interpolate import CubicSpline

from schema.scene import AffineMatrix, Point2D, SceneElement, TextElement
from synthetic.schema import ElementClass

if TYPE_CHECKING:
    from synthetic.config import RandomizationConfig
    from synthetic.loaders.base import LaneSegmentData

_IDENTITY_AFFINE = AffineMatrix(values=[[1, 0, 0], [0, 1, 0], [0, 0, 1]])

_id_counter: int = 0


def _reset_id_counter() -> None:
    global _id_counter
    _id_counter = 0


def _new_id() -> str:
    global _id_counter
    _id_counter += 1
    return f"s{_id_counter:07d}"


def _pts_to_np(pts: list[Point2D]) -> np.ndarray:
    return np.array([[p.x, p.y] for p in pts], dtype=np.float64)


def _np_to_pts(arr: np.ndarray) -> list[Point2D]:
    return [Point2D(x=float(r[0]), y=float(r[1])) for r in arr]


def _arc_resample(pts: np.ndarray, n: int) -> np.ndarray:
    """Resample a polyline to exactly n points by arc-length interpolation."""
    if pts.shape[0] < 2:
        return np.tile(pts[0] if pts.shape[0] > 0 else np.zeros(2), (n, 1))
    diffs = np.diff(pts, axis=0)
    seg_lens = np.sqrt((diffs ** 2).sum(axis=1))
    cum = np.concatenate([[0.0], np.cumsum(seg_lens)])
    total = cum[-1]
    if total < 1e-9:
        return np.tile(pts[0], (n, 1))
    t_query = np.linspace(0.0, total, n)
    out = np.zeros((n, 2), dtype=np.float64)
    for dim in range(2):
        out[:, dim] = np.interp(t_query, cum, pts[:, dim])
    return out


def _bbox(pts: list[Point2D]) -> tuple[float, float, float, float]:
    if not pts:
        return (0.0, 0.0, 0.0, 0.0)
    xs = [p.x for p in pts]
    ys = [p.y for p in pts]
    return (min(xs), min(ys), max(xs), max(ys))


def _make_element(
    pts: np.ndarray,
    elem_class: ElementClass,
    is_dashed: bool = False,
    is_closed: bool = False,
    element_type: str = "polyline",
    layer: str = "LANE_MARKINGS",
    color: str = "black",
    line_width: float = 1.0,
    faro_item_name: str | None = None,
) -> SceneElement:
    pt_list = _np_to_pts(pts)
    return SceneElement(
        id=_new_id(),
        faro_item_name=faro_item_name,
        layer=layer,
        element_type=element_type,
        control_points=pt_list,
        bezier_handles=[],
        interpolation_method="passthrough",
        resampled_points=pt_list,
        transform=_IDENTITY_AFFINE,
        is_closed=is_closed,
        is_dashed=is_dashed,
        line_width=line_width,
        color=color,
        bbox=_bbox(pt_list),
    )


# ---------------------------------------------------------------------------
# Module 1 — drop_primitives
# ---------------------------------------------------------------------------

def drop_primitives(
    lane_marking_elements: list[tuple[SceneElement, ElementClass]],
    stop_line_elements: list[tuple[SceneElement, ElementClass]],
    crosswalk_elements: list[tuple[SceneElement, ElementClass]],
    shoulder_elements: list[tuple[SceneElement, ElementClass]],
    rng: np.random.Generator,
    cfg: "RandomizationConfig",
) -> tuple[
    list[tuple[SceneElement, ElementClass]],
    list[tuple[SceneElement, ElementClass]],
    list[tuple[SceneElement, ElementClass]],
    list[tuple[SceneElement, ElementClass]],
    set[str],
]:
    """
    §1.3 Module 1 — Primitive presence.
    Drops elements by independent per-element Bernoulli draws.
    Returns (filtered lane_markings, stop_lines, crosswalks, shoulders, dropped_ids).
    """
    dropped: set[str] = set()

    def _filter(items: list[tuple[SceneElement, ElementClass]], p: float):
        kept, removed = [], set()
        for elem, cls in items:
            if rng.random() < p:
                removed.add(elem.id)
            else:
                kept.append((elem, cls))
        return kept, removed

    lm, d1 = _filter(lane_marking_elements, cfg.p_drop_lane_markings)
    sl, d2 = _filter(stop_line_elements, cfg.p_drop_stop_lines)
    cw, d3 = _filter(crosswalk_elements, cfg.p_drop_crosswalk_stripes)
    sh, d4 = _filter(shoulder_elements, cfg.p_drop_shoulder_lines)
    dropped = d1 | d2 | d3 | d4
    return lm, sl, cw, sh, dropped


# ---------------------------------------------------------------------------
# Module 2 — resample_vertices
# ---------------------------------------------------------------------------

def resample_vertices(
    elem: SceneElement,
    rng: np.random.Generator,
    cfg: "RandomizationConfig",
) -> SceneElement:
    """
    §1.3 Module 2 — Vertex density and spline conversion.
    Resamples the element's resampled_points to a random count in [vertex_count_min,
    vertex_count_max]. Optionally converts to a cubic not-a-knot polycurve representation.
    """
    if rng.random() >= cfg.p_resample_vertices:
        return elem

    pts = _pts_to_np(elem.resampled_points)
    if pts.shape[0] < 2:
        return elem

    target_n = int(rng.integers(cfg.vertex_count_min, cfg.vertex_count_max + 1))
    resampled = _arc_resample(pts, target_n)
    new_pts = _np_to_pts(resampled)

    as_polycurve = rng.random() < cfg.p_convert_to_polycurve

    if as_polycurve and target_n >= 3:
        t = np.arange(target_n, dtype=np.float64)
        cs_x = CubicSpline(t, resampled[:, 0], bc_type="not-a-knot")
        cs_y = CubicSpline(t, resampled[:, 1], bc_type="not-a-knot")

        # Dense evaluation for resampled_points (20 pts per unit of arc_len, min 20)
        arc_len = float(np.sqrt(np.sum(np.diff(resampled, axis=0) ** 2, axis=1)).sum())
        n_dense = max(20, int(arc_len * 20))
        t_dense = np.linspace(0, target_n - 1, n_dense)
        dense_xy = np.stack([cs_x(t_dense), cs_y(t_dense)], axis=1)
        dense_pts = _np_to_pts(dense_xy)

        # Bezier handles: derivative at each control point scaled to cubic convention
        handles_xy_fwd = np.stack([cs_x(t, 1), cs_y(t, 1)], axis=1) / 3.0
        bezier_handles = _np_to_pts(resampled + handles_xy_fwd)

        return elem.model_copy(update=dict(
            element_type="polycurve",
            interpolation_method="cubic_bezier",
            control_points=new_pts,
            bezier_handles=bezier_handles,
            resampled_points=dense_pts,
            bbox=_bbox(dense_pts),
        ))
    else:
        return elem.model_copy(update=dict(
            element_type="polyline",
            interpolation_method="passthrough",
            control_points=new_pts,
            bezier_handles=[],
            resampled_points=new_pts,
            bbox=_bbox(new_pts),
        ))


# ---------------------------------------------------------------------------
# Module 3 — jitter_vertices
# ---------------------------------------------------------------------------

def jitter_vertices(
    elem: SceneElement,
    rng: np.random.Generator,
    cfg: "RandomizationConfig",
) -> SceneElement:
    """
    §1.3 Module 3 — Vertex jitter (surveyor noise).
    Perturbs interior vertices (endpoints are left unchanged to preserve connectivity).
    When p_jitter_perpendicular_only fires, jitter is applied only in the direction
    normal to the local tangent.
    """
    if rng.random() >= cfg.p_jitter_vertices:
        return elem

    pts = _pts_to_np(elem.resampled_points)
    if pts.shape[0] < 3:
        return elem

    arc_len = float(np.sqrt(np.sum(np.diff(pts, axis=0) ** 2, axis=1)).sum())
    sigma = min(cfg.jitter_sigma_max, cfg.jitter_sigma_max * arc_len / 100.0)

    perp_only = rng.random() < cfg.p_jitter_perpendicular_only
    jittered = pts.copy()

    for i in range(1, pts.shape[0] - 1):  # skip endpoints
        tangent = pts[min(i + 1, pts.shape[0] - 1)] - pts[max(i - 1, 0)]
        t_len = np.linalg.norm(tangent)
        if t_len < 1e-9:
            continue
        tangent /= t_len
        normal = np.array([-tangent[1], tangent[0]])

        noise_mag = float(rng.normal(0.0, sigma))
        if perp_only:
            jittered[i] += noise_mag * normal
        else:
            noise_dir = rng.standard_normal(2)
            noise_dir /= max(np.linalg.norm(noise_dir), 1e-9)
            jittered[i] += noise_mag * noise_dir

    new_pts = _np_to_pts(jittered)
    return elem.model_copy(update={"resampled_points": new_pts, "bbox": _bbox(new_pts)})


# ---------------------------------------------------------------------------
# Module 4 — vary_crosswalk
# ---------------------------------------------------------------------------

def vary_crosswalk(
    road_axis_start: np.ndarray,
    road_axis_end: np.ndarray,
    road_width_m: float,
    rng: np.random.Generator,
    cfg: "RandomizationConfig",
) -> list[tuple[SceneElement, ElementClass]]:
    """
    §1.3 Module 4 — Crosswalk variation.
    Generates a crosswalk as a list of CROSSWALK_STRIPE elements from scratch.
    The road axis defines the along-road direction; stripes run perpendicular to it.
    """
    axis = road_axis_end - road_axis_start
    axis_len = np.linalg.norm(axis)
    if axis_len < 1e-6:
        return []
    axis_unit = axis / axis_len
    normal = np.array([-axis_unit[1], axis_unit[0]])

    stripe_count = int(rng.integers(cfg.crosswalk_stripe_count_min, cfg.crosswalk_stripe_count_max + 1))
    spacing = float(rng.uniform(cfg.crosswalk_spacing_min_m, cfg.crosswalk_spacing_max_m))
    stripe_len = float(rng.uniform(cfg.crosswalk_stripe_length_min_m, cfg.crosswalk_stripe_length_max_m))
    angle_noise = float(rng.uniform(-cfg.crosswalk_angle_noise_deg, cfg.crosswalk_angle_noise_deg))
    angle_rad = math.radians(angle_noise)

    c, s = math.cos(angle_rad), math.sin(angle_rad)
    stripe_dir = np.array([
        normal[0] * c - normal[1] * s,
        normal[0] * s + normal[1] * c,
    ])

    # Total stripe span along axis
    total_span = stripe_count * spacing
    start_along = axis_len / 2.0 - total_span / 2.0

    # Partial occlusion: drop stripes from each end
    drop_start = int(rng.integers(0, cfg.crosswalk_max_occluded_stripes + 1))
    drop_end = int(rng.integers(0, cfg.crosswalk_max_occluded_stripes + 1))
    visible_indices = list(range(drop_start, stripe_count - drop_end))

    elements: list[tuple[SceneElement, ElementClass]] = []
    origin = road_axis_start

    for i in visible_indices:
        along = start_along + i * spacing
        center = origin + along * axis_unit
        p0 = center - stripe_dir * stripe_len / 2.0
        p1 = center + stripe_dir * stripe_len / 2.0
        pts = np.stack([p0, p1])
        elem = _make_element(
            pts, ElementClass.CROSSWALK_STRIPE,
            is_dashed=False, layer="ROADWAY", color="white", line_width=0.5,
        )
        elements.append((elem, ElementClass.CROSSWALK_STRIPE))
    return elements


# ---------------------------------------------------------------------------
# Module 5 — add_annotation_clutter
# ---------------------------------------------------------------------------

def add_annotation_clutter(
    scene_bbox: tuple[float, float, float, float],
    rng: np.random.Generator,
    cfg: "RandomizationConfig",
) -> tuple[list[tuple[SceneElement, ElementClass]], list[TextElement]]:
    """
    §1.3 Module 5 — Annotation clutter.
    Generates reference point markers, grade callouts, north arrow, scale bar,
    and dimension callouts as SceneElements and TextElements.
    """
    if rng.random() >= cfg.p_add_annotation_clutter:
        return [], []

    minx, miny, maxx, maxy = scene_bbox
    w, h = maxx - minx, maxy - miny

    new_elems: list[tuple[SceneElement, ElementClass]] = []
    new_texts: list[TextElement] = []

    def _rand_pos() -> np.ndarray:
        return np.array([
            float(rng.uniform(minx + 0.02 * w, maxx - 0.02 * w)),
            float(rng.uniform(miny + 0.02 * h, maxy - 0.02 * h)),
        ])

    # RP markers
    n_rp = int(rng.integers(0, cfg.clutter_max_rp_markers + 1))
    for i in range(n_rp):
        pos = _rand_pos()
        # Small cross symbol
        cross_pts = np.array([
            [pos[0] - 0.01, pos[1]], [pos[0] + 0.01, pos[1]],
            [pos[0], pos[1] - 0.01], [pos[0], pos[1] + 0.01],
        ])
        elem = _make_element(cross_pts[:2], ElementClass.ANNOTATION_GEOMETRY, layer="ANNOTATIONS", color="black")
        new_elems.append((elem, ElementClass.ANNOTATION_GEOMETRY))
        new_texts.append(TextElement(
            id=_new_id(), text=f"RP{i + 1}",
            position=Point2D(x=float(pos[0]) + 0.012, y=float(pos[1])),
            font_size=8.0,
        ))

    # Grade callouts
    n_grade = int(rng.integers(0, cfg.clutter_max_grade_callouts + 1))
    grade_options = ["+2% Grade", "+1% Grade", "0% Grade", "-1% Grade", "-2% Grade"]
    for _ in range(n_grade):
        pos = _rand_pos()
        new_texts.append(TextElement(
            id=_new_id(),
            text=rng.choice(grade_options),
            position=Point2D(x=float(pos[0]), y=float(pos[1])),
            font_size=7.0,
        ))

    # North arrow
    if rng.random() < cfg.p_clutter_north_arrow:
        origin = np.array([maxx - 0.08 * w, miny + 0.05 * h])
        angle_noise = float(rng.uniform(-0.26, 0.26))  # ±15°
        tip = origin + np.array([math.sin(angle_noise), math.cos(angle_noise)]) * 0.04 * h
        shaft = np.stack([origin, tip])
        elem = _make_element(shaft, ElementClass.ANNOTATION_GEOMETRY, layer="ANNOTATIONS", color="black", line_width=1.5)
        new_elems.append((elem, ElementClass.ANNOTATION_GEOMETRY))
        new_texts.append(TextElement(
            id=_new_id(), text="N",
            position=Point2D(x=float(tip[0]), y=float(tip[1]) + 0.01 * h),
            font_size=9.0,
        ))

    # Scale bar
    if rng.random() < cfg.p_clutter_scale_bar:
        scale_m = rng.choice([15.0, 30.0, 50.0, 100.0])
        bar_len = scale_m / (max(w, h) + 1e-9) * w  # approximate normalized length
        bar_start = np.array([minx + 0.05 * w, miny + 0.03 * h])
        bar_end = bar_start + np.array([bar_len, 0.0])
        bar_pts = np.stack([bar_start, bar_end])
        elem = _make_element(bar_pts, ElementClass.ANNOTATION_GEOMETRY, layer="ANNOTATIONS", color="black", line_width=2.0)
        new_elems.append((elem, ElementClass.ANNOTATION_GEOMETRY))
        new_texts.append(TextElement(
            id=_new_id(), text=f"{scale_m:.0f}m",
            position=Point2D(x=float((bar_start[0] + bar_end[0]) / 2), y=float(bar_start[1]) - 0.02 * h),
            font_size=7.0,
        ))

    return new_elems, new_texts


# ---------------------------------------------------------------------------
# Module 6 — subset_network
# ---------------------------------------------------------------------------

def subset_network(
    segments: "list[LaneSegmentData]",
    focal_point: np.ndarray,
    rng: np.random.Generator,
    cfg: "RandomizationConfig",
) -> tuple["list[LaneSegmentData]", np.ndarray]:
    """
    §1.3 Module 6 — Road network subsetting.
    Crops segments to a random box around focal_point. Returns (surviving_segments, crop_bbox).
    crop_bbox is (minx, miny, maxx, maxy) in source metric coords.
    """
    crop_size = float(rng.uniform(cfg.crop_size_min_m, cfg.crop_size_max_m))
    half = crop_size / 2.0

    if rng.random() > cfg.p_show_full_intersection:
        # Partial crop: shift focal to one approach arm
        shift_dir = rng.standard_normal(2)
        shift_dir /= max(np.linalg.norm(shift_dir), 1e-9)
        focal_point = focal_point + shift_dir * crop_size * 0.25

    minx = focal_point[0] - half
    maxx = focal_point[0] + half
    miny = focal_point[1] - half
    maxy = focal_point[1] + half

    surviving: list = []
    for seg in segments:
        cl = seg.centerline_xy
        if cl.shape[0] == 0:
            continue
        # Keep if any centerline point falls inside crop_bbox
        in_box = (
            (cl[:, 0] >= minx) & (cl[:, 0] <= maxx) &
            (cl[:, 1] >= miny) & (cl[:, 1] <= maxy)
        )
        if in_box.any():
            surviving.append(seg)

    crop_bbox = np.array([minx, miny, maxx, maxy])
    return surviving, crop_bbox


# ---------------------------------------------------------------------------
# Module 7 — transform_layout
# ---------------------------------------------------------------------------

def transform_layout(
    rng: np.random.Generator,
    cfg: "RandomizationConfig",
) -> np.ndarray:
    """
    §1.3 Module 7 — Layout transformation.
    Returns a 3×3 affine applied BEFORE normalization (in source metric coords).
    Composes: random rotation + random scale + optional aspect-ratio skew.
    Note: the north arrow in add_annotation_clutter must subtract this rotation.
    """
    M = np.eye(3, dtype=np.float64)

    if rng.random() < cfg.p_random_rotation:
        angle = float(rng.uniform(0.0, 2 * math.pi))
        c, s = math.cos(angle), math.sin(angle)
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)
        M = R @ M

    if rng.random() < cfg.p_random_scale:
        scale = float(rng.uniform(cfg.layout_scale_min, cfg.layout_scale_max))
        S = np.diag([scale, scale, 1.0])
        M = S @ M

    if rng.random() < cfg.p_random_aspect_ratio:
        aspect = float(rng.uniform(0.6, 1.6))
        A = np.diag([aspect, 1.0 / aspect, 1.0])
        M = A @ M

    return M


def _apply_layout_to_segments(
    segments: "list[LaneSegmentData]",
    M: np.ndarray,
) -> "list[LaneSegmentData]":
    """Apply a pre-normalization layout transform to all segment polylines."""
    import dataclasses

    def _xfm(pts: np.ndarray) -> np.ndarray:
        if pts.shape[0] == 0:
            return pts
        h = np.ones((pts.shape[0], 3))
        h[:, :2] = pts
        return (M @ h.T).T[:, :2]

    result = []
    for seg in segments:
        result.append(dataclasses.replace(
            seg,
            centerline_xy=_xfm(seg.centerline_xy),
            left_boundary_xy=_xfm(seg.left_boundary_xy),
            right_boundary_xy=_xfm(seg.right_boundary_xy),
        ))
    return result


# ---------------------------------------------------------------------------
# Module 8 — add_vehicle_markings
# ---------------------------------------------------------------------------

def add_vehicle_markings(
    segments: "list[LaneSegmentData]",
    rng: np.random.Generator,
    cfg: "RandomizationConfig",
) -> list[tuple[SceneElement, ElementClass]]:
    """
    §1.3 Module 8 — Vehicle and impact markings.
    Adds 1–4 vehicle outline rectangles along sampled lane segments plus optional
    impact markers (X symbols, skid marks). These are distractors for road topology.
    """
    elements: list[tuple[SceneElement, ElementClass]] = []
    if not segments:
        return elements

    n_vehicles = int(rng.integers(cfg.vehicle_count_min, cfg.vehicle_count_max + 1))

    vehicle_w = 2.0   # metres
    vehicle_l = 4.5   # metres

    for _ in range(n_vehicles):
        seg = segments[rng.integers(0, len(segments))]
        cl = seg.centerline_xy
        if cl.shape[0] < 2:
            continue

        # Sample a point along the centerline
        t = float(rng.uniform(0.2, 0.8))
        idx = int(t * (cl.shape[0] - 1))
        center = cl[idx]

        tangent = cl[min(idx + 1, cl.shape[0] - 1)] - cl[max(idx - 1, 0)]
        t_len = np.linalg.norm(tangent)
        if t_len < 1e-6:
            continue
        tangent /= t_len
        normal = np.array([-tangent[1], tangent[0]])

        # Rectangle corners (closed)
        corners = np.array([
            center + tangent * vehicle_l / 2 + normal * vehicle_w / 2,
            center + tangent * vehicle_l / 2 - normal * vehicle_w / 2,
            center - tangent * vehicle_l / 2 - normal * vehicle_w / 2,
            center - tangent * vehicle_l / 2 + normal * vehicle_w / 2,
            center + tangent * vehicle_l / 2 + normal * vehicle_w / 2,  # close
        ])
        elem = _make_element(
            corners, ElementClass.VEHICLE_OUTLINE,
            is_closed=True, layer="VEHICLES", color="black", line_width=1.2,
        )
        elements.append((elem, ElementClass.VEHICLE_OUTLINE))

        # Optional impact markers
        if rng.random() < cfg.p_add_impact_markers:
            # X symbol at vehicle center
            arm = 1.0
            cross1 = np.array([center + np.array([-arm, -arm]), center + np.array([arm, arm])])
            cross2 = np.array([center + np.array([arm, -arm]), center + np.array([-arm, arm])])
            for c_pts in [cross1, cross2]:
                elem = _make_element(
                    c_pts, ElementClass.IMPACT_MARKER, layer="ANNOTATIONS", color="black",
                )
                elements.append((elem, ElementClass.IMPACT_MARKER))

            # Skid marks (3–6 short dashes leading into vehicle)
            n_skid = int(rng.integers(3, 7))
            skid_spacing = 1.5
            for k in range(n_skid):
                offset = (k + 1) * skid_spacing
                skid_center = center - tangent * (vehicle_l / 2 + offset)
                skid_pts = np.array([
                    skid_center + normal * vehicle_w * 0.3,
                    skid_center - normal * vehicle_w * 0.3,
                ])
                elem = _make_element(
                    skid_pts, ElementClass.IMPACT_MARKER,
                    is_dashed=True, layer="ANNOTATIONS", color="black", line_width=0.8,
                )
                elements.append((elem, ElementClass.IMPACT_MARKER))

    return elements


# ---------------------------------------------------------------------------
# Module 9 — apply_faro_styling
# ---------------------------------------------------------------------------

_COLOR_BY_CLASS: dict[ElementClass, tuple[str, ...]] = {
    ElementClass.ROAD_EDGE: ("black",),
    ElementClass.LANE_MARKING_SOLID: ("white", "white", "white", "white", "white",
                                      "white", "white", "white", "white", "yellow"),
    ElementClass.LANE_MARKING_DASHED: ("white", "white", "white", "white", "white",
                                       "white", "white", "white", "yellow", "yellow"),
    ElementClass.STOP_LINE: ("white",),
    ElementClass.SHOULDER_LINE: ("white",),
    ElementClass.CROSSWALK_STRIPE: ("white",),
    ElementClass.ANNOTATION_GEOMETRY: ("black",),
    ElementClass.ANNOTATION_TEXT: ("black",),
    ElementClass.VEHICLE_OUTLINE: ("black",),
    ElementClass.IMPACT_MARKER: ("black",),
    ElementClass.SIGN_SYMBOL: ("black",),
    ElementClass.ARROW_SYMBOL: ("white",),
    ElementClass.OTHER: ("black",),
}


def apply_faro_styling(
    elem: SceneElement,
    elem_class: ElementClass,
    rng: np.random.Generator,
    cfg: "RandomizationConfig",
) -> SceneElement:
    """
    §1.3 Module 9 — FARO-specific stroke styling.
    Randomizes line_width and color within FARO conventions per element class.
    is_dashed is preserved from the element's existing value (set during construction).
    """
    if rng.random() >= cfg.p_apply_faro_styling:
        return elem

    line_width = float(rng.uniform(cfg.faro_line_weight_min, cfg.faro_line_weight_max))

    color_choices = _COLOR_BY_CLASS.get(elem_class, ("black",))
    color = str(rng.choice(color_choices))

    return elem.model_copy(update={"line_width": line_width, "color": color})

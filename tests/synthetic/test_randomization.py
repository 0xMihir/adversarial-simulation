import numpy as np
import pytest

from schema.scene import AffineMatrix, Point2D, SceneElement
from synthetic.config import RandomizationConfig
from synthetic.randomization import (
    apply_faro_styling,
    drop_primitives,
    jitter_vertices,
    resample_vertices,
    vary_crosswalk,
)
from synthetic.schema import ElementClass

_IDENTITY = AffineMatrix(values=[[1, 0, 0], [0, 1, 0], [0, 0, 1]])


def _pts(coords: list[tuple[float, float]]) -> list[Point2D]:
    return [Point2D(x=x, y=y) for x, y in coords]


def _elem(coords: list[tuple[float, float]], eid: str = "e1", is_dashed: bool = False) -> SceneElement:
    pt_list = _pts(coords)
    xs, ys = [p.x for p in pt_list], [p.y for p in pt_list]
    return SceneElement(
        id=eid,
        element_type="polyline",
        control_points=pt_list,
        interpolation_method="passthrough",
        resampled_points=pt_list,
        transform=_IDENTITY,
        is_closed=False,
        is_dashed=is_dashed,
        bbox=(min(xs), min(ys), max(xs), max(ys)),
    )


# ---- drop_primitives ----

def test_drop_primitives_p1_drops_all_lane_markings():
    cfg = RandomizationConfig(p_drop_lane_markings=1.0)
    rng = np.random.default_rng(0)
    elems = [(_elem([(0, 0), (1, 0)], f"e{i}"), ElementClass.LANE_MARKING_SOLID) for i in range(5)]
    surviving, _, _, _, dropped = drop_primitives(elems, [], [], [], rng, cfg)
    assert surviving == []
    assert len(dropped) == 5


def test_drop_primitives_p0_drops_nothing():
    cfg = RandomizationConfig(p_drop_lane_markings=0.0)
    rng = np.random.default_rng(0)
    elems = [(_elem([(0, 0), (1, 0)], f"e{i}"), ElementClass.LANE_MARKING_SOLID) for i in range(5)]
    surviving, _, _, _, dropped = drop_primitives(elems, [], [], [], rng, cfg)
    assert len(surviving) == 5
    assert len(dropped) == 0


# ---- resample_vertices ----

def test_resample_vertices_count_in_range():
    cfg = RandomizationConfig(p_resample_vertices=1.0, vertex_count_min=4, vertex_count_max=20)
    coords = [(float(i), 0.0) for i in range(50)]
    elem = _elem(coords)
    for seed in range(20):
        rng = np.random.default_rng(seed)
        result = resample_vertices(elem, rng, cfg)
        n = len(result.resampled_points)
        assert 4 <= n <= 20, f"seed={seed}: got {n} points"


def test_resample_vertices_polycurve_conversion():
    cfg = RandomizationConfig(p_resample_vertices=1.0, p_convert_to_polycurve=1.0, vertex_count_min=5, vertex_count_max=5)
    coords = [(float(i), float(i * 0.5)) for i in range(20)]
    elem = _elem(coords)
    rng = np.random.default_rng(0)
    result = resample_vertices(elem, rng, cfg)
    assert result.element_type == "polycurve"
    assert result.interpolation_method == "cubic_bezier"
    assert len(result.bezier_handles) > 0


def test_resample_vertices_deterministic():
    cfg = RandomizationConfig(p_resample_vertices=1.0)
    coords = [(float(i), 0.0) for i in range(30)]
    elem = _elem(coords)
    r1 = resample_vertices(elem, np.random.default_rng(99), cfg)
    r2 = resample_vertices(elem, np.random.default_rng(99), cfg)
    assert len(r1.resampled_points) == len(r2.resampled_points)
    for a, b in zip(r1.resampled_points, r2.resampled_points):
        assert abs(a.x - b.x) < 1e-9


# ---- jitter_vertices ----

def test_jitter_vertices_endpoints_unchanged():
    cfg = RandomizationConfig(p_jitter_vertices=1.0, jitter_sigma_max=5.0, p_jitter_perpendicular_only=0.0)
    coords = [(float(i), 0.0) for i in range(20)]
    elem = _elem(coords)
    rng = np.random.default_rng(7)
    result = jitter_vertices(elem, rng, cfg)
    # Endpoints must be identical
    assert abs(result.resampled_points[0].x - coords[0][0]) < 1e-9
    assert abs(result.resampled_points[0].y - coords[0][1]) < 1e-9
    assert abs(result.resampled_points[-1].x - coords[-1][0]) < 1e-9
    assert abs(result.resampled_points[-1].y - coords[-1][1]) < 1e-9


def test_jitter_vertices_no_fire_when_p0():
    cfg = RandomizationConfig(p_jitter_vertices=0.0)
    coords = [(float(i), 0.0) for i in range(10)]
    elem = _elem(coords)
    result = jitter_vertices(elem, np.random.default_rng(0), cfg)
    for orig, new in zip(coords, result.resampled_points):
        assert abs(orig[0] - new.x) < 1e-9 and abs(orig[1] - new.y) < 1e-9


# ---- vary_crosswalk ----

def test_vary_crosswalk_returns_crosswalk_stripes():
    cfg = RandomizationConfig(
        crosswalk_stripe_count_min=4, crosswalk_stripe_count_max=10,
        crosswalk_max_occluded_stripes=0,
    )
    rng = np.random.default_rng(3)
    road_start = np.array([0.0, 0.0])
    road_end = np.array([20.0, 0.0])
    elems = vary_crosswalk(road_start, road_end, road_width_m=6.0, rng=rng, cfg=cfg)
    assert len(elems) >= 4
    for e, cls in elems:
        assert cls == ElementClass.CROSSWALK_STRIPE
        assert len(e.resampled_points) == 2


# ---- apply_faro_styling ----

def test_apply_faro_styling_road_edge_always_black():
    cfg = RandomizationConfig(p_apply_faro_styling=1.0)
    elem = _elem([(0, 0), (1, 0)])
    for seed in range(10):
        rng = np.random.default_rng(seed)
        result = apply_faro_styling(elem, ElementClass.ROAD_EDGE, rng, cfg)
        assert result.color == "black", f"seed={seed}: ROAD_EDGE color should be black"


def test_apply_faro_styling_no_fire_when_p0():
    cfg = RandomizationConfig(p_apply_faro_styling=0.0)
    elem = _elem([(0, 0), (1, 0)])
    elem_with_color = SceneElement(**{**elem.model_dump(), "color": "purple", "line_width": 99.0})
    result = apply_faro_styling(elem_with_color, ElementClass.ROAD_EDGE, np.random.default_rng(0), cfg)
    assert result.color == "purple"
    assert result.line_width == 99.0


def test_apply_faro_styling_deterministic():
    cfg = RandomizationConfig(p_apply_faro_styling=1.0)
    elem = _elem([(0, 0), (1, 0)])
    r1 = apply_faro_styling(elem, ElementClass.LANE_MARKING_SOLID, np.random.default_rng(5), cfg)
    r2 = apply_faro_styling(elem, ElementClass.LANE_MARKING_SOLID, np.random.default_rng(5), cfg)
    assert r1.color == r2.color
    assert r1.line_width == r2.line_width

import numpy as np
from schema.scene import AffineMatrix, ParsedScene, Point2D, SceneElement

from synthetic.normalization import DiagramNormalizer, _apply_2d, _pts_to_np

_IDENTITY = AffineMatrix(values=[[1, 0, 0], [0, 1, 0], [0, 0, 1]])


def _make_elem(pts: list[tuple[float, float]], elem_id: str = "e1") -> SceneElement:
    pt_list = [Point2D(x=x, y=y) for x, y in pts]
    xs, ys = [p.x for p in pt_list], [p.y for p in pt_list]
    return SceneElement(
        id=elem_id,
        element_type="polyline",
        control_points=pt_list,
        interpolation_method="passthrough",
        resampled_points=pt_list,
        transform=_IDENTITY,
        is_closed=False,
        is_dashed=False,
        bbox=(min(xs), min(ys), max(xs), max(ys)),
    )


def _make_scene(pts: list[tuple[float, float]]) -> ParsedScene:
    elem = _make_elem(pts)
    return ParsedScene(
        case_id="test",
        far_filename="test.far",
        parsed_at="2025-01-01T00:00:00Z",
        elements=[elem],
        texts=[],
        vehicles=[],
        roadway_indices=[0],
        road_marking_indices=[],
        other_indices=[],
    )


def test_normalize_roundtrip():
    scene = _make_scene([(0, 0), (10, 0), (10, 5), (0, 5)])
    norm_scene, M_inv = DiagramNormalizer.normalize(scene, apply_pca=False)

    # Collect normalized vertices
    norm_pts = _pts_to_np(norm_scene.elements[0].resampled_points)

    # Apply inverse: should approximately recover original
    orig_pts = _pts_to_np(scene.elements[0].resampled_points)
    recovered = _apply_2d(M_inv, norm_pts)
    np.testing.assert_allclose(recovered, orig_pts, atol=1e-6)


def test_normalize_max_vertex_distance():
    """After normalization, all vertices should be within ±0.5 of origin (unit diagonal)."""
    scene = _make_scene([(0, 0), (100, 0), (100, 50), (0, 50)])
    norm_scene, _ = DiagramNormalizer.normalize(scene, apply_pca=False)
    pts = _pts_to_np(norm_scene.elements[0].resampled_points)
    # The longest diagonal should be ~1.0, so all vertices within radius 0.5 of centroid
    assert np.abs(pts).max() <= 0.6  # some margin for non-square shapes


def test_normalize_pca_false_means_zero_rotation():
    scene = _make_scene([(0, 0), (10, 3), (20, 6)])
    _, M_inv = DiagramNormalizer.normalize(scene, apply_pca=False)
    # Extract rotation angle from M_inv (should be ~0)
    # M_inv = T_inv @ R_inv @ S_inv
    # For pca_angle=0, R_inv = identity → M_inv[0,1] and M_inv[1,0] ≈ 0
    assert abs(M_inv[0, 1]) < 1e-9
    assert abs(M_inv[1, 0]) < 1e-9


def test_normalize_degenerate_single_element():
    """Should not raise for a degenerate scene with only 1 point."""
    scene = _make_scene([(5.0, 5.0), (5.0, 5.0)])
    norm_scene, M_inv = DiagramNormalizer.normalize(scene, apply_pca=False)
    assert norm_scene is not None


def test_compute_normalization_params_pca_disabled():
    vertices = np.array([[0, 0], [10, 3], [20, 6]], dtype=np.float64)
    centroid, pca_angle, scale = DiagramNormalizer.compute_normalization_params(vertices, apply_pca=False)
    assert pca_angle == 0.0
    assert scale > 0


def test_collect_all_vertices_empty():
    # Build a scene with no elements directly rather than via _make_scene([])
    scene = ParsedScene(
        case_id="test", far_filename="test.far", parsed_at="2025-01-01T00:00:00Z",
        elements=[], texts=[], vehicles=[],
        roadway_indices=[], road_marking_indices=[], other_indices=[],
    )
    verts = DiagramNormalizer.collect_all_vertices(scene)
    assert verts.shape == (0, 2)

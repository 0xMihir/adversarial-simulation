"""Unit tests for models.data_transform.lane_graph."""
import numpy as np
import pytest

from models.data_transform.lane_graph import (
    MERGE_EPS,
    SPLIT_INTERVAL,
    LaneGraph,
    _fit_quadratic_ctrl,
    build_lane_graph,
    empty_lane_graph,
)


def _line(p0, p1, n=5) -> np.ndarray:
    t = np.linspace(0.0, 1.0, n)[:, None]
    return np.asarray(p0, float) + t * (np.asarray(p1, float) - np.asarray(p0, float))


def _edge_set(g: LaneGraph) -> set[tuple[int, int]]:
    return set(zip(g.edge_src.tolist(), g.edge_dst.tolist()))


def test_empty_input():
    g = build_lane_graph({}, {}, {})
    assert g.nodes.shape == (0, 2)
    assert g.edge_src.shape == (0,)
    assert g.edge_ctrl.shape == (0, 2)


def test_y_junction_merge():
    # A and B both flow into C: end(A) == end(B) == start(C) is one junction node.
    cls = {
        "A": _line([-0.15, 0.05], [0.0, 0.0]),
        "B": _line([-0.15, -0.05], [0.0, 0.0]),
        "C": _line([0.0, 0.0], [0.15, 0.0]),
    }
    entry = {"A": [], "B": [], "C": ["A", "B"]}
    exit_ = {"A": ["C"], "B": ["C"], "C": []}
    g = build_lane_graph(cls, entry, exit_)
    # nodes: start(A), start(B), junction, end(C) — junction has in-degree 2
    assert g.nodes.shape[0] == 4
    assert g.edge_src.shape[0] == 3
    indeg = np.bincount(g.edge_dst, minlength=g.nodes.shape[0])
    assert indeg.max() == 2
    junction = int(indeg.argmax())
    np.testing.assert_allclose(g.nodes[junction], [0.0, 0.0], atol=1e-9)


def test_proximity_fallback_merge():
    # No topology links, but endpoints coincide within MERGE_EPS -> still merged.
    off = MERGE_EPS * 0.5
    cls = {
        "A": _line([-0.15, 0.0], [0.0, 0.0]),
        "B": _line([0.0, off], [0.15, 0.0]),
    }
    g = build_lane_graph(cls, {}, {})
    assert g.nodes.shape[0] == 3  # merged shared endpoint


def test_no_merge_beyond_eps():
    cls = {
        "A": _line([-0.15, 0.0], [0.0, 0.0]),
        "B": _line([0.0, 10 * MERGE_EPS], [0.15, 0.05]),
    }
    g = build_lane_graph(cls, {}, {})
    assert g.nodes.shape[0] == 4


def test_passthrough_collapse_and_resplit():
    # Chain A->B->C of collinear short lanes: interior degree-(1,1) nodes collapse,
    # then the merged edge (length 0.45 > SPLIT_INTERVAL) re-splits into ceil(0.45/0.2)=3
    # equal-arclength edges.
    cls = {
        "A": _line([0.0, 0.0], [0.15, 0.0]),
        "B": _line([0.15, 0.0], [0.30, 0.0]),
        "C": _line([0.30, 0.0], [0.45, 0.0]),
    }
    entry = {"A": [], "B": ["A"], "C": ["B"]}
    exit_ = {"A": ["B"], "B": ["C"], "C": []}
    g = build_lane_graph(cls, entry, exit_)
    k = int(np.ceil(0.45 / SPLIT_INTERVAL))
    assert g.edge_src.shape[0] == k
    assert g.nodes.shape[0] == k + 1
    # equal-arclength split points on the straight line
    xs = np.sort(g.nodes[:, 0])
    np.testing.assert_allclose(xs, np.linspace(0.0, 0.45, k + 1), atol=1e-9)


def test_short_chain_collapses_to_single_edge():
    cls = {
        "A": _line([0.0, 0.0], [0.05, 0.0]),
        "B": _line([0.05, 0.0], [0.10, 0.0]),
    }
    entry = {"A": [], "B": ["A"]}
    exit_ = {"A": ["B"], "B": []}
    g = build_lane_graph(cls, entry, exit_)
    assert g.nodes.shape[0] == 2
    assert g.edge_src.shape[0] == 1


def test_junction_not_collapsed():
    # Fork: A -> B and A -> C. end(A) has out-degree 2 — must remain a node.
    cls = {
        "A": _line([-0.15, 0.0], [0.0, 0.0]),
        "B": _line([0.0, 0.0], [0.15, 0.05]),
        "C": _line([0.0, 0.0], [0.15, -0.05]),
    }
    entry = {"A": [], "B": ["A"], "C": ["A"]}
    exit_ = {"A": ["B", "C"], "B": [], "C": []}
    g = build_lane_graph(cls, entry, exit_)
    assert g.nodes.shape[0] == 4
    assert g.edge_src.shape[0] == 3


def test_pure_cycle_terminates():
    # Directed triangle of degree-(1,1) nodes: must not infinite-loop; collapses to a
    # single anchor node with one self-loop edge (possibly re-split by length).
    p = [np.array([0.0, 0.0]), np.array([0.1, 0.0]), np.array([0.05, 0.1])]
    cls = {
        "A": _line(p[0], p[1]),
        "B": _line(p[1], p[2]),
        "C": _line(p[2], p[0]),
    }
    entry = {"A": ["C"], "B": ["A"], "C": ["B"]}
    exit_ = {"A": ["B"], "B": ["C"], "C": ["A"]}
    g = build_lane_graph(cls, entry, exit_)
    assert g.edge_src.shape[0] >= 1
    # every node reachable; total edges form the cycle
    assert g.nodes.shape[0] == g.edge_src.shape[0] or (
        g.nodes.shape[0] == 1 and g.edge_src.shape[0] == 1
    )


def test_bezier_fit_roundtrip():
    p0 = np.array([0.0, 0.0])
    p1 = np.array([0.25, 0.4])  # ground-truth middle control point
    p2 = np.array([0.5, 0.0])
    t = np.linspace(0, 1, 50)[:, None]
    poly = (1 - t) ** 2 * p0 + 2 * t * (1 - t) * p1 + t**2 * p2
    # default (fast) setting must land within half a ctrl quantization bin (~4.4e-3)
    fit = _fit_quadratic_ctrl(poly, p0, p2)
    np.testing.assert_allclose(fit, p1, atol=2e-3)
    # with more refinement it converges tightly
    fit6 = _fit_quadratic_ctrl(poly, p0, p2, iters=6, grid=129)
    np.testing.assert_allclose(fit6, p1, atol=1e-4)


def test_bezier_fit_two_point_fallback():
    p0 = np.array([0.0, 0.0])
    p2 = np.array([0.1, 0.0])
    fit = _fit_quadratic_ctrl(np.stack([p0, p2]), p0, p2)
    np.testing.assert_allclose(fit, 0.5 * (p0 + p2), atol=1e-12)


def test_dfs_deterministic():
    cls = {
        "A": _line([-0.15, 0.05], [0.0, 0.0]),
        "B": _line([-0.15, -0.05], [0.0, 0.0]),
        "C": _line([0.0, 0.0], [0.15, 0.0]),
    }
    entry = {"A": [], "B": [], "C": ["A", "B"]}
    exit_ = {"A": ["C"], "B": ["C"], "C": []}
    g1 = build_lane_graph(cls, entry, exit_)
    g2 = build_lane_graph(dict(reversed(cls.items())), entry, exit_)
    np.testing.assert_allclose(g1.nodes, g2.nodes)
    assert _edge_set(g1) == _edge_set(g2)


def test_missing_topology_ids_ignored():
    cls = {"A": _line([0.0, 0.0], [0.1, 0.0])}
    entry = {"A": ["GHOST"]}
    exit_ = {"A": ["ALSO_GHOST"]}
    g = build_lane_graph(cls, entry, exit_)
    assert g.edge_src.shape[0] == 1


def test_degenerate_lane_dropped():
    cls = {"A": np.array([[0.0, 0.0]]), "B": np.zeros((3, 2))}
    g = build_lane_graph(cls, {}, {})
    assert g.nodes.shape[0] == 0


def test_empty_lane_graph_helper():
    g = empty_lane_graph()
    assert g.nodes.shape == (0, 2)

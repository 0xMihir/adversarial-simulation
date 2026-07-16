"""Round-trip and robustness tests for the SeqGrowGraph tokenizer."""
import numpy as np

from models.data_transform.lane_graph import LaneGraph, empty_lane_graph
from models.seq_grow_graph.tokenizer import decode_tokens, encode_graph
from models.seq_grow_graph.vocab import (
    COORD_MAX,
    COORD_MIN,
    CTRL_MAX,
    CTRL_MIN,
    MAX_NODES,
    N_COORD_BINS,
    N_CTRL_BINS,
    NODE_INDEX_BASE,
    TOK_BOS,
    TOK_EOS,
    TOK_SEP,
    TOK_TO,
)

COORD_HALF_BIN = 0.5 * (COORD_MAX - COORD_MIN) / N_COORD_BINS
CTRL_HALF_BIN = 0.5 * (CTRL_MAX - CTRL_MIN) / N_CTRL_BINS


def _graph(nodes, edges) -> LaneGraph:
    """edges: list of (src, dst, cx, cy)."""
    return LaneGraph(
        nodes=np.asarray(nodes, np.float64).reshape(-1, 2),
        edge_src=np.asarray([e[0] for e in edges], np.int64),
        edge_dst=np.asarray([e[1] for e in edges], np.int64),
        edge_ctrl=np.asarray([[e[2], e[3]] for e in edges], np.float64).reshape(-1, 2),
    )


def _edge_dict(g: LaneGraph) -> dict[tuple[int, int], np.ndarray]:
    return {
        (int(s), int(d)): c
        for s, d, c in zip(g.edge_src, g.edge_dst, g.edge_ctrl)
    }


def _assert_roundtrip(g: LaneGraph):
    dec = decode_tokens(encode_graph(g))
    assert dec.nodes.shape == g.nodes.shape
    np.testing.assert_allclose(dec.nodes, g.nodes, atol=COORD_HALF_BIN + 1e-12)
    ge, de = _edge_dict(g), _edge_dict(dec)
    assert set(ge) == set(de)
    for k in ge:
        np.testing.assert_allclose(de[k], ge[k], atol=CTRL_HALF_BIN + 1e-12)


def test_empty_graph():
    toks = encode_graph(empty_lane_graph())
    assert toks.tolist() == [TOK_BOS, TOK_EOS]
    dec = decode_tokens(toks)
    assert dec.nodes.shape[0] == 0


def test_two_node_layout_by_hand():
    # node1 introduces the edge 0->1 in its from-list
    g = _graph([[-0.1, 0.0], [0.1, 0.05]], [(0, 1, 0.0, 0.2)])
    toks = encode_graph(g).tolist()
    assert toks[0] == TOK_BOS and toks[-1] == TOK_EOS
    # node 0: x, y, TO, SEP (no edges yet)
    assert toks[3] == TOK_TO and toks[4] == TOK_SEP
    # node 1: x, y, idx0, cx, cy, TO, SEP
    assert toks[7] == NODE_INDEX_BASE + 0
    assert toks[10] == TOK_TO and toks[11] == TOK_SEP
    assert len(toks) == 13


def test_roundtrip_y_junction():
    g = _graph(
        [[-0.3, 0.1], [-0.3, -0.1], [0.0, 0.0], [0.3, 0.0]],
        [(0, 2, -0.15, 0.05), (1, 2, -0.15, -0.05), (2, 3, 0.15, 0.0)],
    )
    _assert_roundtrip(g)


def test_roundtrip_back_edge_and_self_loop():
    # edge 2->0 (to-list back-reference) and self-loop 1->1
    g = _graph(
        [[0.0, 0.0], [0.2, 0.0], [0.4, 0.1]],
        [(0, 1, 0.1, 0.0), (1, 1, 0.25, 0.3), (2, 0, 0.2, 0.4)],
    )
    _assert_roundtrip(g)


def test_roundtrip_random_graphs():
    rng = np.random.default_rng(7)
    for _ in range(20):
        n = int(rng.integers(1, 30))
        nodes = rng.uniform(-0.6, 0.6, (n, 2))
        n_e = int(rng.integers(0, 3 * n))
        edges = set()
        while len(edges) < n_e:
            edges.add((int(rng.integers(0, n)), int(rng.integers(0, n))))
        g = _graph(nodes, [(s, d, *rng.uniform(-0.9, 0.9, 2)) for s, d in edges])
        _assert_roundtrip(g)


def test_node_capacity_truncation():
    n = MAX_NODES + 20
    nodes = np.stack([np.linspace(-0.6, 0.6, n), np.zeros(n)], axis=1)
    edges = [(i, i + 1, 0.0, 0.0) for i in range(n - 1)]
    g = _graph(nodes, edges)
    toks = encode_graph(g, max_seq_len=10_000)
    dec = decode_tokens(toks)
    assert dec.nodes.shape[0] == MAX_NODES
    assert dec.edge_src.max() < MAX_NODES and dec.edge_dst.max() < MAX_NODES


def test_seq_len_truncation_at_sep_boundary():
    n = 60
    nodes = np.stack([np.linspace(-0.6, 0.6, n), np.zeros(n)], axis=1)
    g = _graph(nodes, [(i, i + 1, 0.0, 0.0) for i in range(n - 1)])
    toks = encode_graph(g, max_seq_len=100)
    assert len(toks) <= 100
    assert toks[-1] == TOK_EOS and toks[-2] == TOK_SEP  # cut on a subsequence boundary
    dec = decode_tokens(toks)  # valid prefix graph
    assert 0 < dec.nodes.shape[0] < n
    # chain prefix: every kept node except the first has its incoming edge
    assert dec.edge_src.shape[0] == dec.nodes.shape[0] - 1


def test_tolerant_decode_garbage():
    rng = np.random.default_rng(3)
    garbage = rng.integers(0, 575, 300)
    decode_tokens(garbage)  # must not raise
    # truncated mid-triple, missing EOS
    g = _graph([[0.0, 0.0], [0.1, 0.0]], [(0, 1, 0.05, 0.0)])
    toks = encode_graph(g)[:-3]
    decode_tokens(toks)  # must not raise


def test_decode_recovers_after_missing_sep():
    # node0 sub-sequence missing SEP, immediately followed by node1 coords
    x0, y0 = 10, 20
    x1, y1 = 30, 40
    toks = np.array([TOK_BOS, x0, y0, TOK_TO, x1, y1, TOK_TO, TOK_SEP, TOK_EOS])
    dec = decode_tokens(toks)
    assert dec.nodes.shape[0] == 2

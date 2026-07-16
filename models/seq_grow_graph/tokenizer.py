"""
LaneGraph <-> SeqGrowGraph token sequence.

Serialization (paper Fig. 4, <From>/<None> markers omitted): nodes are already in DFS
order; introducing node n emits

  x_bin, y_bin,
  [for each existing k with edge k->n, ascending k:  NODE_INDEX_BASE+k, ctrl_x, ctrl_y],
  <TO>,
  [for each existing q with edge n->q, ascending q:  NODE_INDEX_BASE+q, ctrl_x, ctrl_y],
  <SEP>

wrapped in <BOS> ... <EOS>. Every edge appears exactly once — in the subsequence of
its later-indexed endpoint (self-loops n->n appear in node n's to-list).

Truncation: graphs with more than MAX_NODES nodes keep only the first MAX_NODES DFS
nodes and the edges among them (a DFS prefix is a self-consistent subgraph); sequences
that would exceed MAX_SEQ_LEN are cut at the last complete <SEP> boundary. Scenes are
truncated, never dropped, so training still sees topologically rich scenes.

decode_tokens is deliberately tolerant (skips malformed stretches) because it also
consumes raw model output at inference time.
"""
from __future__ import annotations

import numpy as np

from models.data_transform.lane_graph import LaneGraph, empty_lane_graph
from models.seq_grow_graph.vocab import (
    CTRL_BASE,
    MAX_NODES,
    MAX_SEQ_LEN,
    N_COORD_BINS,
    N_CTRL_BINS,
    NODE_INDEX_BASE,
    TOK_BOS,
    TOK_EOS,
    TOK_PAD,
    TOK_SEP,
    TOK_TO,
    bin_to_coord,
    bin_to_ctrl,
    coord_to_bin,
    ctrl_to_bin,
)


def encode_graph(graph: LaneGraph, max_seq_len: int = MAX_SEQ_LEN) -> np.ndarray:
    """Serialize a DFS-ordered LaneGraph into a (T,) int64 token sequence."""
    n = graph.nodes.shape[0]
    if n > MAX_NODES:
        keep = graph.edge_src < MAX_NODES
        keep &= graph.edge_dst < MAX_NODES
        graph = LaneGraph(
            nodes=graph.nodes[:MAX_NODES],
            edge_src=graph.edge_src[keep],
            edge_dst=graph.edge_dst[keep],
            edge_ctrl=graph.edge_ctrl[keep],
        )
        n = MAX_NODES

    # group edges by their later-indexed endpoint
    from_lists: list[list[tuple[int, np.ndarray]]] = [[] for _ in range(n)]  # k -> node
    to_lists: list[list[tuple[int, np.ndarray]]] = [[] for _ in range(n)]  # node -> q
    for s, d, c in zip(graph.edge_src, graph.edge_dst, graph.edge_ctrl):
        s, d = int(s), int(d)
        if d >= s:  # incl. self-loop: emitted at node d(==s) in the to-list of...
            if s == d:
                to_lists[d].append((s, c))
            else:
                from_lists[d].append((s, c))
        else:
            to_lists[s].append((d, c))

    tokens: list[int] = [TOK_BOS]
    for i in range(n):
        xb = int(coord_to_bin(graph.nodes[i, 0]))
        yb = int(coord_to_bin(graph.nodes[i, 1]))
        sub: list[int] = [xb, yb]
        for k, c in sorted(from_lists[i], key=lambda e: e[0]):
            sub += [NODE_INDEX_BASE + k, int(ctrl_to_bin(c[0])), int(ctrl_to_bin(c[1]))]
        sub.append(TOK_TO)
        for q, c in sorted(to_lists[i], key=lambda e: e[0]):
            sub += [NODE_INDEX_BASE + q, int(ctrl_to_bin(c[0])), int(ctrl_to_bin(c[1]))]
        sub.append(TOK_SEP)
        if len(tokens) + len(sub) + 1 > max_seq_len:  # +1 for EOS
            break
        tokens.extend(sub)
    tokens.append(TOK_EOS)
    return np.asarray(tokens, dtype=np.int64)


def _is_coord(t: int) -> bool:
    return 0 <= t < N_COORD_BINS


def _is_index(t: int) -> bool:
    return NODE_INDEX_BASE <= t < NODE_INDEX_BASE + MAX_NODES


def _is_ctrl(t: int) -> bool:
    return CTRL_BASE <= t < CTRL_BASE + N_CTRL_BINS


def decode_tokens(tokens: np.ndarray) -> LaneGraph:
    """Parse a token sequence (GT or model output) back into a LaneGraph.

    Tolerant state machine: stops at <EOS>/<PAD>, skips tokens that don't fit the
    grammar, drops edge triples that reference not-yet-introduced nodes. Node
    positions / ctrl points come back as bin centers.
    """
    toks = [int(t) for t in np.asarray(tokens).ravel()]
    nodes: list[tuple[float, float]] = []
    edges: list[tuple[int, int, float, float]] = []

    i = 0
    end = len(toks)
    while i < end and toks[i] in (TOK_BOS, TOK_PAD):
        i += 1
    while i < end:
        t = toks[i]
        if t in (TOK_EOS, TOK_PAD):
            break
        # ---- node coords ----
        if not (_is_coord(t) and i + 1 < end and _is_coord(toks[i + 1])):
            i += 1  # garbage where a node should start; resync
            continue
        x = float(bin_to_coord(t))
        y = float(bin_to_coord(toks[i + 1]))
        i += 2
        cur = len(nodes)
        nodes.append((x, y))
        # ---- from-list, <TO>, to-list, <SEP> ----
        mode = "from"
        while i < end:
            t = toks[i]
            if t in (TOK_EOS, TOK_PAD):
                break
            if t == TOK_TO:
                mode = "to"
                i += 1
                continue
            if t == TOK_SEP:
                i += 1
                break
            if _is_coord(t):
                break  # model skipped <SEP>; treat as the next node starting
            if (
                _is_index(t)
                and i + 2 < end
                and _is_ctrl(toks[i + 1])
                and _is_ctrl(toks[i + 2])
            ):
                other = t - NODE_INDEX_BASE
                cx = float(bin_to_ctrl(toks[i + 1]))
                cy = float(bin_to_ctrl(toks[i + 2]))
                if other <= cur:
                    if mode == "from":
                        edges.append((other, cur, cx, cy))
                    else:
                        edges.append((cur, other, cx, cy))
                i += 3
                continue
            i += 1  # malformed token inside a subsequence; skip

    if not nodes:
        return empty_lane_graph()
    return LaneGraph(
        nodes=np.asarray(nodes, dtype=np.float64).reshape(-1, 2),
        edge_src=np.asarray([e[0] for e in edges], dtype=np.int64),
        edge_dst=np.asarray([e[1] for e in edges], dtype=np.int64),
        edge_ctrl=np.asarray([[e[2], e[3]] for e in edges], dtype=np.float64).reshape(-1, 2),
    )

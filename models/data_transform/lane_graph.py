"""
SyntheticGroundTruth lane centerlines + topology -> SeqGrowGraph lane graph.

SeqGrowGraph (arXiv 2507.04822) represents the lane network as a directed graph
G = (V, E): V are junction / key topological nodes, E are centerlines between them,
each summarized as a quadratic Bezier whose endpoints are the node positions and whose
middle control point is fit to the centerline polyline.

This module is pure numpy (no torch / pydantic) so it can run inside DataLoader
workers. Inputs are plain dicts of arrays/lists; a thin adapter converts from
SyntheticGroundTruth at the call site (models.data_transform.scene_to_point).

Pipeline (paper section 3.1 + 4.5):
  1. merge lane endpoints into nodes (successor/predecessor links + proximity fallback)
  2. one directed edge per lane
  3. collapse degree-(1,1) pass-through nodes, splicing centerlines
  4. re-split edges longer than SPLIT_INTERVAL at equal arclength
  5. fit a quadratic Bezier middle control point per edge
  6. relabel nodes in deterministic DFS order
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np

# Coincident-endpoint merge radius in normalized scene units (~0.5m at ~100m scene
# extent). Successor/predecessor links are the primary merge signal; this catches
# endpoint pairs whose topology link was pruned.
MERGE_EPS = 5e-3
# Re-split edges longer than this arclength (normalized units). Scenes are normalized
# so the longest hull diagonal = 1.0 (~100m-ish), so 0.20 ~ 20m — the low end of the
# paper's 20-40m sweet spot, keeping the quadratic-Bezier fit error small.
SPLIT_INTERVAL = 0.20

_EPS = 1e-12


class LaneGraph(NamedTuple):
    """Directed lane graph, nodes already in DFS order."""

    nodes: np.ndarray  # (n, 2) float64 node positions
    edge_src: np.ndarray  # (E,) int64
    edge_dst: np.ndarray  # (E,) int64
    edge_ctrl: np.ndarray  # (E, 2) float64 quadratic-Bezier middle control points


def empty_lane_graph() -> LaneGraph:
    return LaneGraph(
        nodes=np.zeros((0, 2), np.float64),
        edge_src=np.zeros((0,), np.int64),
        edge_dst=np.zeros((0,), np.int64),
        edge_ctrl=np.zeros((0, 2), np.float64),
    )


class _UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, a: int) -> int:
        while self.parent[a] != a:
            self.parent[a] = self.parent[self.parent[a]]
            a = self.parent[a]
        return a

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def _arclength(poly: np.ndarray) -> float:
    if poly.shape[0] < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(poly, axis=0), axis=1).sum())


def _solve_ctrl(q: np.ndarray, tj: np.ndarray, p0: np.ndarray, p2: np.ndarray) -> np.ndarray | None:
    """Closed-form least-squares middle ctrl point for fixed parameters tj."""
    w = 2.0 * tj * (1.0 - tj)
    denom = float((w * w).sum())
    if denom < _EPS:
        return None
    resid = q - np.outer((1.0 - tj) ** 2, p0) - np.outer(tj**2, p2)
    return (w[:, None] * resid).sum(axis=0) / denom


def _fit_quadratic_ctrl(
    poly: np.ndarray, p0: np.ndarray, p2: np.ndarray, iters: int = 2, grid: int = 65
) -> np.ndarray:
    """Least-squares middle control point of a quadratic Bezier through `poly`.

    Endpoints are fixed at p0/p2 (node positions). Initial chord-length
    parameterization, then `iters` rounds of parameter re-estimation by projecting
    each polyline point onto the current curve (nearest of `grid` samples + parabolic
    refinement) — removes the chord-vs-curve parameter bias on curved edges. The
    defaults land ~0.25 ctrl-quantization-half-bins of error on a worst-case curvy
    edge at ~130us/edge; more iterations buy precision the tokenizer bins can't see.
    Falls back to the endpoint midpoint when there are no usable interior points.
    """
    mid = 0.5 * (p0 + p2)
    if poly.shape[0] < 3:
        return mid
    seg_len = np.linalg.norm(np.diff(poly, axis=0), axis=1)
    total = seg_len.sum()
    if total < _EPS:
        return mid
    t = np.concatenate([[0.0], np.cumsum(seg_len)]) / total  # (m,)
    q = poly[1:-1]
    tj = t[1:-1]
    p1 = _solve_ctrl(q, tj, p0, p2)
    if p1 is None:
        return mid

    ts = np.linspace(0.0, 1.0, grid)
    for _ in range(iters):
        curve = (
            np.outer((1.0 - ts) ** 2, p0)
            + np.outer(2.0 * ts * (1.0 - ts), p1)
            + np.outer(ts**2, p2)
        )  # (S, 2)
        d2 = ((q[:, None, :] - curve[None, :, :]) ** 2).sum(-1)  # (m-2, S)
        idx = d2.argmin(axis=1)
        # parabolic refinement of the squared-distance minimum over the sample grid
        i0 = np.clip(idx, 1, len(ts) - 2)
        y0, y1, y2 = d2[np.arange(len(idx)), i0 - 1], d2[np.arange(len(idx)), i0], d2[
            np.arange(len(idx)), i0 + 1
        ]
        denom = y0 - 2 * y1 + y2
        shift = np.where(np.abs(denom) > _EPS, 0.5 * (y0 - y2) / np.maximum(denom, _EPS), 0.0)
        tj_new = np.clip(ts[i0] + np.clip(shift, -1.0, 1.0) * (ts[1] - ts[0]), 0.0, 1.0)
        p1_new = _solve_ctrl(q, tj_new, p0, p2)
        if p1_new is None:
            break
        if np.linalg.norm(p1_new - p1) < 1e-12:
            p1 = p1_new
            break
        p1 = p1_new
    return p1


def _split_polyline(poly: np.ndarray, k: int) -> list[np.ndarray]:
    """Split a polyline into k pieces of equal arclength, interpolating cut points."""
    seg_len = np.linalg.norm(np.diff(poly, axis=0), axis=1)
    total = seg_len.sum()
    if k <= 1 or total < _EPS:
        return [poly]
    cum = np.concatenate([[0.0], np.cumsum(seg_len)])
    cuts = [total * j / k for j in range(1, k)]
    pieces: list[np.ndarray] = []
    cur: list[np.ndarray] = [poly[0]]
    ci = 0  # next cut index
    for i in range(1, poly.shape[0]):
        while ci < len(cuts) and cuts[ci] <= cum[i] + _EPS:
            s = cuts[ci]
            if seg_len[i - 1] < _EPS:
                cut_pt = poly[i]
            else:
                alpha = (s - cum[i - 1]) / seg_len[i - 1]
                alpha = min(max(alpha, 0.0), 1.0)
                cut_pt = poly[i - 1] + alpha * (poly[i] - poly[i - 1])
            cur.append(cut_pt)
            pieces.append(np.array(cur))
            cur = [cut_pt]
            ci += 1
        cur.append(poly[i])
    pieces.append(np.array(cur))
    return pieces


def build_lane_graph(
    centerlines: dict[str, np.ndarray],
    entry_ids: dict[str, list[str]],
    exit_ids: dict[str, list[str]],
) -> LaneGraph:
    """Build the SeqGrowGraph lane graph from lane centerlines + connectivity.

    Args:
        centerlines: lane_id -> (M, 2) float polyline in normalized scene coords.
        entry_ids: lane_id -> predecessor lane ids (their END meets this lane's START).
        exit_ids: lane_id -> successor lane ids (this lane's END meets their START).

    Ids present in topology but missing from `centerlines` are ignored (topology is
    pruned to surviving lanes, but degenerate <2-point centerlines can still be absent).
    """
    lane_ids = [
        lid
        for lid, pts in centerlines.items()
        if np.asarray(pts).shape[0] >= 2 and _arclength(np.asarray(pts, np.float64)) > _EPS
    ]
    if not lane_ids:
        return empty_lane_graph()
    polys = {lid: np.asarray(centerlines[lid], dtype=np.float64) for lid in lane_ids}
    lane_pos = {lid: i for i, lid in enumerate(lane_ids)}

    # --- 1. endpoint merge: endpoint 2i = start of lane i, 2i+1 = end of lane i ---
    n_lanes = len(lane_ids)
    uf = _UnionFind(2 * n_lanes)
    for lid in lane_ids:
        i = lane_pos[lid]
        for suc in exit_ids.get(lid, []):
            if suc in lane_pos:
                uf.union(2 * i + 1, 2 * lane_pos[suc])
        for pre in entry_ids.get(lid, []):
            if pre in lane_pos:
                uf.union(2 * i, 2 * lane_pos[pre] + 1)

    endpoint_xy = np.empty((2 * n_lanes, 2), np.float64)
    for lid in lane_ids:
        i = lane_pos[lid]
        endpoint_xy[2 * i] = polys[lid][0]
        endpoint_xy[2 * i + 1] = polys[lid][-1]

    # proximity fallback (lane counts are small; brute force is fine)
    for a in range(2 * n_lanes):
        for b in range(a + 1, 2 * n_lanes):
            if uf.find(a) != uf.find(b) and (
                np.linalg.norm(endpoint_xy[a] - endpoint_xy[b]) <= MERGE_EPS
            ):
                uf.union(a, b)

    root_to_node: dict[int, int] = {}
    node_members: list[list[int]] = []
    endpoint_node = np.empty(2 * n_lanes, np.int64)
    for e in range(2 * n_lanes):
        r = uf.find(e)
        if r not in root_to_node:
            root_to_node[r] = len(node_members)
            node_members.append([])
        node_members[root_to_node[r]].append(e)
        endpoint_node[e] = root_to_node[r]
    node_xy = np.array([endpoint_xy[m].mean(axis=0) for m in node_members])

    # --- 2. one directed edge per lane: (src_node, dst_node, polyline) ---
    edges: list[tuple[int, int, np.ndarray]] = []
    for lid in lane_ids:
        i = lane_pos[lid]
        src, dst = int(endpoint_node[2 * i]), int(endpoint_node[2 * i + 1])
        if src == dst and _arclength(polys[lid]) < MERGE_EPS:
            continue
        edges.append((src, dst, polys[lid]))
    if not edges:
        return empty_lane_graph()

    # --- 3. collapse degree-(1,1) pass-through nodes ---
    changed = True
    while changed:
        changed = False
        in_edges: dict[int, list[int]] = {}
        out_edges: dict[int, list[int]] = {}
        for ei, (s, d, _) in enumerate(edges):
            out_edges.setdefault(s, []).append(ei)
            in_edges.setdefault(d, []).append(ei)
        for v in list(in_edges.keys()):
            ins, outs = in_edges.get(v, []), out_edges.get(v, [])
            if len(ins) != 1 or len(outs) != 1 or ins[0] == outs[0]:
                continue
            e_in, e_out = ins[0], outs[0]
            s, _, poly_in = edges[e_in]
            _, d, poly_out = edges[e_out]
            spliced = np.concatenate([poly_in, poly_out[1:]], axis=0)
            edges[e_in] = (s, d, spliced)
            edges.pop(e_out)
            changed = True
            break  # adjacency maps are stale; rebuild

    # --- 4. re-split long edges, inserting nodes at cut points ---
    node_list: list[np.ndarray] = [node_xy[i] for i in range(node_xy.shape[0])]
    split_edges: list[tuple[int, int, np.ndarray]] = []
    for s, d, poly in edges:
        length = _arclength(poly)
        k = max(1, int(np.ceil(length / SPLIT_INTERVAL))) if length > SPLIT_INTERVAL else 1
        pieces = _split_polyline(poly, k)
        cur = s
        for j, piece in enumerate(pieces):
            if j < len(pieces) - 1:
                node_list.append(piece[-1].copy())
                nxt = len(node_list) - 1
            else:
                nxt = d
            split_edges.append((cur, nxt, piece))
            cur = nxt
    edges = split_edges
    node_xy = np.array(node_list)

    # --- 5. Bezier fit + drop nodes orphaned by the collapse ---
    used = sorted({s for s, _, _ in edges} | {d for _, d, _ in edges})
    remap = {old: new for new, old in enumerate(used)}
    node_xy = node_xy[used]
    fitted = [
        (remap[s], remap[d], _fit_quadratic_ctrl(poly, node_xy[remap[s]], node_xy[remap[d]]))
        for s, d, poly in edges
    ]

    # --- 6. deterministic DFS relabel ---
    n = node_xy.shape[0]
    indeg = np.zeros(n, np.int64)
    undirected: dict[int, set[int]] = {i: set() for i in range(n)}
    for s, d, _ in fitted:
        indeg[d] += 1
        undirected[s].add(d)
        undirected[d].add(s)

    def _key(i: int) -> tuple[float, float]:
        return (float(node_xy[i, 0]), float(node_xy[i, 1]))

    order: list[int] = []
    visited = np.zeros(n, bool)
    roots = sorted((i for i in range(n) if indeg[i] == 0), key=_key)
    roots += sorted((i for i in range(n) if indeg[i] > 0), key=_key)
    for root in roots:
        if visited[root]:
            continue
        stack = [root]
        while stack:
            v = stack.pop()
            if visited[v]:
                continue
            visited[v] = True
            order.append(v)
            for nb in sorted(undirected[v], key=_key, reverse=True):
                if not visited[nb]:
                    stack.append(nb)
    dfs_rank = np.empty(n, np.int64)
    for rank, old in enumerate(order):
        dfs_rank[old] = rank

    edge_src = np.array([dfs_rank[s] for s, _, _ in fitted], np.int64)
    edge_dst = np.array([dfs_rank[d] for _, d, _ in fitted], np.int64)
    edge_ctrl = np.array([c for _, _, c in fitted], np.float64).reshape(-1, 2)
    return LaneGraph(
        nodes=node_xy[order],
        edge_src=edge_src,
        edge_dst=edge_dst,
        edge_ctrl=edge_ctrl,
    )

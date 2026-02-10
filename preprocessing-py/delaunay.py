import heapq
import numpy as np
from scipy.spatial import Delaunay
from scipy.linalg import norm
from scipy.spatial import KDTree
from collections import deque
from pyclothoids import Clothoid

# Try to import numba for JIT compilation, fall back to no-op decorator if unavailable
try:
    from numba import njit, prange

    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

    # Create no-op decorators as fallback
    def njit(*args, **kwargs):
        def decorator(func):
            return func

        if len(args) == 1 and callable(args[0]):
            return args[0]
        return decorator

    def prange(*args):
        return range(*args)

# ============================================================================
# NUMBA-ACCELERATED CORE FUNCTIONS
# ============================================================================


@njit(cache=True)
def ccw_fast(ax, ay, bx, by, cx, cy):
    """JIT-compiled counter-clockwise test."""
    return (cy - ay) * (bx - ax) > (by - ay) * (cx - ax)


@njit(cache=True)
def line_segments_intersect_fast(ax, ay, bx, by, cx, cy, dx, dy):
    """JIT-compiled line segment intersection check."""
    return ccw_fast(ax, ay, cx, cy, dx, dy) != ccw_fast(
        bx, by, cx, cy, dx, dy
    ) and ccw_fast(ax, ay, bx, by, cx, cy) != ccw_fast(ax, ay, bx, by, dx, dy)


@njit(cache=True)
def compute_edge_angle_fast(start_x, start_y, end_x, end_y):
    """JIT-compiled edge angle computation."""
    return np.arctan2(end_y - start_y, end_x - start_x)


@njit(cache=True)
def angles_are_parallel_fast(angle1, angle2, epsilon_rad):
    """JIT-compiled parallel angle check."""
    diff = np.arctan2(np.sin(angle1 - angle2), np.cos(angle1 - angle2))
    abs_diff = np.abs(diff)
    return abs_diff < epsilon_rad or abs_diff > (np.pi - epsilon_rad)


@njit(cache=True)
def point_to_segment_distance_fast(px, py, ax, ay, bx, by):
    """JIT-compiled point to line segment distance."""
    edge_x = bx - ax
    edge_y = by - ay
    edge_length_sq = edge_x * edge_x + edge_y * edge_y

    if edge_length_sq < 1e-12:  # Degenerate edge
        return np.sqrt((px - ax) ** 2 + (py - ay) ** 2)

    # Project point onto line
    t = max(0.0, min(1.0, ((px - ax) * edge_x + (py - ay) * edge_y) / edge_length_sq))
    closest_x = ax + t * edge_x
    closest_y = ay + t * edge_y

    return np.sqrt((px - closest_x) ** 2 + (py - closest_y) ** 2)


# ============================================================================
# VECTORIZED PREPROCESSING
# ============================================================================


def compute_perpendicular_widths(simplices, all_points, point_to_roadway_idx):
    """
    Compute perpendicular road width for every triangle.

    For triangles with two vertices on the same roadway, the width is the
    distance from the opposite vertex to the shared-roadway edge.  When all
    three vertices belong to different roadways the minimum altitude is used.

    Returns:
        perp_widths: (n_triangles,) array of perpendicular widths
    """
    n = len(simplices)
    perp_widths = np.empty(n)

    for idx in range(n):
        simplex = simplices[idx]
        perp_w = None
        for j in range(3):
            v0, v1, v2 = simplex[j], simplex[(j + 1) % 3], simplex[(j + 2) % 3]
            if point_to_roadway_idx[v0] == point_to_roadway_idx[v1]:
                perp_w = point_to_segment_distance_fast(
                    all_points[v2, 0],
                    all_points[v2, 1],
                    all_points[v0, 0],
                    all_points[v0, 1],
                    all_points[v1, 0],
                    all_points[v1, 1],
                )
                break
        if perp_w is None:
            altitudes = []
            for j in range(3):
                v_opp = simplex[j]
                v_a, v_b = simplex[(j + 1) % 3], simplex[(j + 2) % 3]
                altitudes.append(
                    point_to_segment_distance_fast(
                        all_points[v_opp, 0],
                        all_points[v_opp, 1],
                        all_points[v_a, 0],
                        all_points[v_a, 1],
                        all_points[v_b, 0],
                        all_points[v_b, 1],
                    )
                )
            perp_w = min(altitudes)
        perp_widths[idx] = perp_w

    return perp_widths


def precompute_roadway_edge_angles(roadway_edges):
    """
    Precompute angles for all roadway edges once.

    Returns:
        Dict mapping roadway_idx to numpy array of edge angles
    """
    roadway_edge_angles = {}
    for roadway_idx, edges in roadway_edges.items():
        angles = np.array(
            [compute_edge_angle_fast(e[0][0], e[0][1], e[1][0], e[1][1]) for e in edges]
        )
        roadway_edge_angles[roadway_idx] = angles
    return roadway_edge_angles


def build_roadway_edge_arrays(roadway_edges):
    """
    Convert roadway edges dict to flat numpy arrays for faster access.

    Returns:
        edge_starts: (n_edges, 2) array
        edge_ends: (n_edges, 2) array
        edge_roadway_idx: (n_edges,) array mapping edge to roadway
        roadway_edge_ranges: dict mapping roadway_idx to (start, end) indices
    """
    edge_starts = []
    edge_ends = []
    edge_roadway_idx = []
    roadway_edge_ranges = {}

    current_idx = 0
    for roadway_idx, edges in roadway_edges.items():
        start_idx = current_idx
        for edge_start, edge_end in edges:
            edge_starts.append(edge_start)
            edge_ends.append(edge_end)
            edge_roadway_idx.append(roadway_idx)
            current_idx += 1
        roadway_edge_ranges[roadway_idx] = (start_idx, current_idx)

    return (
        np.array(edge_starts),
        np.array(edge_ends),
        np.array(edge_roadway_idx),
        roadway_edge_ranges,
    )



# ============================================================================
# BATCH PARALLEL EDGE CHECKING
# ============================================================================


def find_nearest_edge_batch(
    points,
    roadway_idx,
    edge_starts,
    edge_ends,
    roadway_edge_ranges,
    search_distance=2.0,
):
    """
    Find nearest edges for multiple points at once.

    Args:
        points: (n, 2) array of query points
        roadway_idx: which roadway to search
        edge_starts, edge_ends: precomputed edge arrays
        roadway_edge_ranges: dict of (start, end) indices per roadway

    Returns:
        nearest_edges: list of (edge_start, edge_end, angle) or None for each point
    """
    if roadway_idx not in roadway_edge_ranges:
        return [None] * len(points)

    start_idx, end_idx = roadway_edge_ranges[roadway_idx]
    edges_start = edge_starts[start_idx:end_idx]
    edges_end = edge_ends[start_idx:end_idx]

    if len(edges_start) == 0:
        return [None] * len(points)

    results = []
    for pt in points:
        best_dist = search_distance
        best_edge = None

        for i in range(len(edges_start)):
            dist = point_to_segment_distance_fast(
                pt[0],
                pt[1],
                edges_start[i, 0],
                edges_start[i, 1],
                edges_end[i, 0],
                edges_end[i, 1],
            )
            if dist < best_dist:
                best_dist = dist
                angle = compute_edge_angle_fast(
                    edges_start[i, 0],
                    edges_start[i, 1],
                    edges_end[i, 0],
                    edges_end[i, 1],
                )
                best_edge = (edges_start[i].copy(), edges_end[i].copy(), angle)

        results.append(best_edge)

    return results


def check_parallel_vectorized(
    simplex_batch,
    all_points,
    point_to_roadway_idx,
    edge_starts,
    edge_ends,
    roadway_edge_ranges,
    parallel_angle_epsilon,
):
    """
    Vectorized parallel edge check for a batch of triangles.

    Returns:
        is_parallel: boolean array for each triangle
    """
    epsilon_rad = np.deg2rad(parallel_angle_epsilon)
    n_triangles = len(simplex_batch)
    is_parallel = np.zeros(n_triangles, dtype=bool)

    for tri_idx, simplex in enumerate(simplex_batch):
        # Get roadway indices for this triangle's vertices
        roadway_indices = [point_to_roadway_idx[idx] for idx in simplex]
        unique_roadways = set(roadway_indices)

        if len(unique_roadways) < 2:
            continue

        # Find nearest edges for each vertex
        vertex_angles = {}  # roadway_idx -> list of angles

        for vert_idx, (vertex, roadway_idx) in enumerate(zip(simplex, roadway_indices)):
            pt = all_points[vertex : vertex + 1]  # Keep 2D shape
            edge_result = find_nearest_edge_batch(
                pt,
                roadway_idx,
                edge_starts,
                edge_ends,
                roadway_edge_ranges,
                search_distance=2.0,
            )[0]

            if edge_result is not None:
                if roadway_idx not in vertex_angles:
                    vertex_angles[roadway_idx] = []
                vertex_angles[roadway_idx].append(edge_result[2])  # angle

        # Check if we have angles from at least 2 different roadways
        if len(vertex_angles) >= 2:
            roadway_mean_angles = {
                ridx: np.mean(angles) for ridx, angles in vertex_angles.items()
            }

            # Check all pairs
            roadway_list = list(roadway_mean_angles.items())
            for i in range(len(roadway_list)):
                for j in range(i + 1, len(roadway_list)):
                    if angles_are_parallel_fast(
                        roadway_list[i][1], roadway_list[j][1], epsilon_rad
                    ):
                        is_parallel[tri_idx] = True
                        break
                if is_parallel[tri_idx]:
                    break

    return is_parallel


# ============================================================================
# MAIN OPTIMIZED FUNCTION
# ============================================================================


def resample_polyline(points, step_distance=2.0):
    """Interpolates points along a polyline so they are spaced no more than step_distance apart."""
    if len(points) < 2:
        return np.array(points)

    new_points = [points[0]]
    for i in range(1, len(points)):
        p1 = np.array(points[i - 1])
        p2 = np.array(points[i])
        dist = np.linalg.norm(p2 - p1)

        if dist > step_distance:
            num_points = int(np.ceil(dist / step_distance))
            interpolated = np.linspace(p1, p2, num_points + 1)[1:]
            new_points.extend(interpolated)
        else:
            new_points.append(p2)

    return np.array(new_points)


def build_edge_spatial_index(roadway_edges):
    """Build R-tree spatial index for fast edge lookup."""
    from rtree import index

    idx = index.Index()
    edge_list = []
    edge_id = 0

    for roadway_idx, edges in roadway_edges.items():
        for edge_start, edge_end in edges:
            bbox = (
                min(edge_start[0], edge_end[0]),
                min(edge_start[1], edge_end[1]),
                max(edge_start[0], edge_end[0]),
                max(edge_start[1], edge_end[1]),
            )
            idx.insert(edge_id, bbox)
            edge_list.append((roadway_idx, edge_start, edge_end))
            edge_id += 1

    return idx, edge_list


def check_crossing_optimized(
    center_curr, center_nbr, edge_index, edge_list, relevant_roadways
):
    """Check if line segment crosses any roadway edges using spatial index."""
    bbox = (
        min(center_curr[0], center_nbr[0]),
        min(center_curr[1], center_nbr[1]),
        max(center_curr[0], center_nbr[0]),
        max(center_curr[1], center_nbr[1]),
    )

    for edge_id in edge_index.intersection(bbox):
        roadway_idx, road_seg_start, road_seg_end = edge_list[edge_id]
        if roadway_idx in relevant_roadways:
            if line_segments_intersect_fast(
                center_curr[0],
                center_curr[1],
                center_nbr[0],
                center_nbr[1],
                road_seg_start[0],
                road_seg_start[1],
                road_seg_end[0],
                road_seg_end[1],
            ):
                return True
    return False


def smooth_segments(segments, window=3):
    """
    Apply moving-average smoothing to each polyline segment.

    Endpoints are preserved so connectivity is maintained.
    Interior points are replaced by the mean of their window-sized neighbourhood.

    Args:
        segments: list of polylines (each a list of (x,y) tuples)
        window: number of neighbours on each side to average (total kernel = 2*window+1)
    Returns:
        Smoothed list of polylines.
    """
    smoothed = []
    for seg in segments:
        pts = np.array(seg)
        n = len(pts)
        if n <= 2 * window + 1:
            smoothed.append(seg)
            continue

        out = pts.copy()
        for i in range(window, n - window):
            out[i] = pts[i - window : i + window + 1].mean(axis=0)

        smoothed.append([tuple(p) for p in out])
    return smoothed


def split_wide_roads(
    centerline_segments,
    midpoint_width_map,
    split_threshold=24.0,
    triple_split_threshold=30.0,
):
    """
    Split centerline polylines where the road is wider than split_threshold.

    For roads wider than triple_split_threshold, generates three offset polylines
    at -width/3, 0, +width/3 from the original centerline.
    For roads wider than split_threshold (but below triple), generates two offset
    polylines at +/- width/4 from the original centerline.
    Handles curved roads by computing per-point normals via finite differences.

    Args:
        centerline_segments: list of polylines (each a list of (x,y) tuples)
        midpoint_width_map: dict mapping (x,y) -> perpendicular road width
        split_threshold: minimum perpendicular width to trigger a 2-way split
        triple_split_threshold: minimum perpendicular width to trigger a 3-way split
    Returns:
        Updated list of polylines with wide roads split.
    """
    result = []

    for polyline in centerline_segments:
        if len(polyline) < 2:
            result.append(polyline)
            continue

        pts = np.array(polyline)

        # Look up perpendicular width for each point
        widths = np.array([midpoint_width_map.get((p[0], p[1]), 0.0) for p in pts])

        # Use median to be robust to occasional narrow pinch points
        median_width = np.median(widths)
        if median_width < split_threshold:
            result.append(polyline)
            continue

        # Compute per-point tangents using finite differences
        tangents = np.zeros_like(pts)
        tangents[1:-1] = pts[2:] - pts[:-2]
        tangents[0] = pts[1] - pts[0]
        tangents[-1] = pts[-1] - pts[-2]

        # Normalize tangents
        lengths = np.linalg.norm(tangents, axis=1, keepdims=True)
        tangents = tangents / np.maximum(lengths, 1e-12)

        # Normals = rotate tangent 90 degrees CCW: (tx, ty) -> (-ty, tx)
        normals = np.column_stack([-tangents[:, 1], tangents[:, 0]])

        if median_width >= triple_split_threshold:
            # 3-way split: lines at -width/3, center, +width/3
            offsets = widths / 3.0
            lane_width = widths / 3.0

            left_pts = pts + normals * offsets[:, np.newaxis]
            right_pts = pts - normals * offsets[:, np.newaxis]

            left_polyline = [tuple(p) for p in left_pts]
            center_polyline = [tuple(p) for p in pts]
            right_polyline = [tuple(p) for p in right_pts]

            for pt, lw in zip(left_polyline, lane_width):
                midpoint_width_map[pt] = lw
            for pt, lw in zip(center_polyline, lane_width):
                midpoint_width_map[pt] = lw
            for pt, lw in zip(right_polyline, lane_width):
                midpoint_width_map[pt] = lw

            result.append(left_polyline)
            result.append(center_polyline)
            result.append(right_polyline)
        else:
            # 2-way split: lines at +/- width/4
            offsets = widths / 4.0
            half_widths = widths / 2.0

            left_pts = pts + normals * offsets[:, np.newaxis]
            right_pts = pts - normals * offsets[:, np.newaxis]

            left_polyline = [tuple(p) for p in left_pts]
            right_polyline = [tuple(p) for p in right_pts]

            for pt, hw in zip(left_polyline, half_widths):
                midpoint_width_map[pt] = hw
            for pt, hw in zip(right_polyline, half_widths):
                midpoint_width_map[pt] = hw

            result.append(left_polyline)
            result.append(right_polyline)

    return result


# ============================================================================
# ENDPOINT CONNECTION WITH CLOTHOID INTERPOLATION
# ============================================================================


def clothoid_interpolate(p0, theta0, p1, theta1, n=10):
    """
    Sample n points along a clothoid from (p0, theta0) to (p1, theta1).

    Falls back to linear interpolation if the clothoid fit fails.

    Returns:
        List of (x, y) tuples (excluding the start/end points themselves).
    """
    try:
        clothoid = Clothoid.G1Hermite(p0[0], p0[1], theta0, p1[0], p1[1], theta1)
        xs, ys = clothoid.SampleXY(n + 2)  # +2 for start/end
        # Exclude first and last (they duplicate the endpoints)
        return [(xs[i], ys[i]) for i in range(1, len(xs) - 1)]
    except Exception:
        # Linear fallback
        interp_x = np.linspace(p0[0], p1[0], n + 2)[1:-1]
        interp_y = np.linspace(p0[1], p1[1], n + 2)[1:-1]
        return [(interp_x[i], interp_y[i]) for i in range(len(interp_x))]


def connect_endpoints_clothoid(polylines, distance_threshold=36.0, angle_threshold=30.0):
    """
    Merge polylines by connecting their endpoints with clothoid curves.

    Only first/last points of each polyline are candidates for connection.
    Connections are made greedily by shortest distance, each endpoint used
    at most once. Connected endpoints are joined with a 10-point clothoid.

    Args:
        polylines: list of polylines, each a list of (x, y) tuples
        distance_threshold: max distance between endpoints to consider connecting
        angle_threshold: max angular deviation (degrees) between endpoint headings

    Returns:
        (merged_polylines, outlier_info)
    """
    if not polylines:
        return [], []

    angle_threshold_rad = np.deg2rad(angle_threshold)

    # Collect endpoints with metadata
    endpoints = []  # [(x, y), ...]
    endpoint_meta = []  # [(polyline_idx, 'start'|'end'), ...]

    for i, seg in enumerate(polylines):
        if len(seg) < 2:
            continue
        endpoints.append(seg[0])
        endpoint_meta.append((i, "start"))
        endpoints.append(seg[-1])
        endpoint_meta.append((i, "end"))

    if len(endpoints) < 2:
        return list(polylines), []

    ep_array = np.array(endpoints)
    tree = KDTree(ep_array)
    pairs = tree.query_pairs(r=distance_threshold)

    # Compute headings for each endpoint (pointing outward from the polyline)
    headings = []
    for i, (poly_idx, end_type) in enumerate(endpoint_meta):
        seg = polylines[poly_idx]
        if end_type == "end":
            headings.append(
                compute_edge_angle_fast(seg[-2][0], seg[-2][1], seg[-1][0], seg[-1][1])
            )
        else:
            headings.append(
                compute_edge_angle_fast(seg[1][0], seg[1][1], seg[0][0], seg[0][1])
            )

    # Score and filter candidate pairs
    candidates = []
    for i, j in pairs:
        poly_i, end_i = endpoint_meta[i]
        poly_j, end_j = endpoint_meta[j]

        if poly_i == poly_j:
            continue

        dist = np.linalg.norm(ep_array[i] - ep_array[j])

        # Check that endpoint headings align with the connection direction
        conn_angle_ij = compute_edge_angle_fast(
            ep_array[i, 0], ep_array[i, 1], ep_array[j, 0], ep_array[j, 1]
        )

        diff_i = np.abs(
            np.arctan2(
                np.sin(headings[i] - conn_angle_ij),
                np.cos(headings[i] - conn_angle_ij),
            )
        )
        conn_angle_ji = conn_angle_ij + np.pi
        diff_j = np.abs(
            np.arctan2(
                np.sin(headings[j] - conn_angle_ji),
                np.cos(headings[j] - conn_angle_ji),
            )
        )

        if diff_i > angle_threshold_rad or diff_j > angle_threshold_rad:
            continue

        candidates.append((dist, i, j))

    # Greedy merge by shortest distance
    candidates.sort()
    used_endpoints = set()
    used_polyline_ends = set()
    merges = []

    for dist, i, j in candidates:
        if i in used_endpoints or j in used_endpoints:
            continue
        poly_i, end_i = endpoint_meta[i]
        poly_j, end_j = endpoint_meta[j]
        if (poly_i, end_i) in used_polyline_ends:
            continue
        if (poly_j, end_j) in used_polyline_ends:
            continue

        used_endpoints.add(i)
        used_endpoints.add(j)
        used_polyline_ends.add((poly_i, end_i))
        used_polyline_ends.add((poly_j, end_j))
        merges.append((i, j))

    # Build merged polylines
    merged_poly_indices = set()
    merged_polylines = []

    for i, j in merges:
        poly_i, end_i = endpoint_meta[i]
        poly_j, end_j = endpoint_meta[j]
        merged_poly_indices.add(poly_i)
        merged_poly_indices.add(poly_j)

        seg_i = list(polylines[poly_i])
        seg_j = list(polylines[poly_j])

        # Orient so the connecting end of seg_i is at the tail
        if end_i == "start":
            seg_i = seg_i[::-1]
        # Orient so the connecting end of seg_j is at the head
        if end_j == "end":
            seg_j = seg_j[::-1]

        # Heading into the connection from seg_i's tail
        theta_i = compute_edge_angle_fast(
            seg_i[-2][0], seg_i[-2][1], seg_i[-1][0], seg_i[-1][1]
        )
        # Heading out of the connection into seg_j's head
        theta_j = compute_edge_angle_fast(
            seg_j[0][0], seg_j[0][1], seg_j[1][0], seg_j[1][1]
        )

        bridge = clothoid_interpolate(seg_i[-1], theta_i, seg_j[0], theta_j, n=10)
        merged_polylines.append(seg_i + bridge + seg_j)

    # Add unmerged polylines
    for idx, seg in enumerate(polylines):
        if idx not in merged_poly_indices:
            merged_polylines.append(list(seg))

    outlier_info = []
    return merged_polylines, outlier_info


# ============================================================================
# BFS CONNECT
# ============================================================================


def bfs_connect_optimized(
    start_idx, adjacency, edges, visited, points, angle_threshold, distance_threshold
):
    """
    BFS using a directional cone to prevent backwards connections.

    A candidate point must lie within a forward-facing cone originating at
    the tip of the current segment.  The cone is *dynamic*: closer points
    are allowed a wider angular deviation while farther points must be
    more tightly aligned.  The constraint is:

        (dist / distance_threshold) + (angle_diff / max_angle) <= 1.0

    Both ``angle_diff <= max_angle`` and ``dist <= distance_threshold``
    are still enforced as hard limits.
    """
    segment = deque([start_idx])
    visited.add(start_idx)
    current = start_idx
    max_angle_rad = np.deg2rad(angle_threshold)

    # Forward search
    while True:
        candidates = []
        for nbr in adjacency[current]:
            if nbr in visited:
                continue
            edge_key = (min(current, nbr), max(current, nbr))
            edge_info = edges.get(edge_key)
            if not edge_info:
                continue

            dist = edge_info["distance"]

            if len(segment) > 1:
                prev = segment[-2]
                fwd_angle = compute_edge_angle_fast(
                    points[prev, 0],
                    points[prev, 1],
                    points[current, 0],
                    points[current, 1],
                )
                nbr_angle = compute_edge_angle_fast(
                    points[current, 0],
                    points[current, 1],
                    points[nbr, 0],
                    points[nbr, 1],
                )
                # Angular deviation from the forward direction
                angle_diff = np.abs(
                    np.arctan2(
                        np.sin(nbr_angle - fwd_angle), np.cos(nbr_angle - fwd_angle)
                    )
                )

                # Hard angle limit
                if angle_diff > max_angle_rad:
                    continue

                # Dynamic cone: linear trade-off between distance and angle
                norm_dist = dist / distance_threshold
                norm_angle = angle_diff / max_angle_rad
                if norm_dist + norm_angle > 1.0:
                    continue

            heapq.heappush(candidates, (dist, nbr))

        if not candidates:
            break

        _, next_idx = heapq.heappop(candidates)
        visited.add(next_idx)
        segment.append(next_idx)
        current = next_idx

    # Backward search from start
    if len(segment) >= 2:
        # Backward direction = opposite of the segment's forward start
        bwd_angle = compute_edge_angle_fast(
            points[segment[1], 0],
            points[segment[1], 1],
            points[segment[0], 0],
            points[segment[0], 1],
        )
        current = segment[0]

        while True:
            candidates = []
            for nbr in adjacency[current]:
                if nbr in visited:
                    continue
                edge_key = (min(current, nbr), max(current, nbr))
                edge_info = edges.get(edge_key)
                if not edge_info:
                    continue

                dist = edge_info["distance"]
                nbr_angle = compute_edge_angle_fast(
                    points[current, 0],
                    points[current, 1],
                    points[nbr, 0],
                    points[nbr, 1],
                )

                # Angular deviation from backward direction
                angle_diff = np.abs(
                    np.arctan2(
                        np.sin(nbr_angle - bwd_angle), np.cos(nbr_angle - bwd_angle)
                    )
                )

                if angle_diff > max_angle_rad:
                    continue

                norm_dist = dist / distance_threshold
                norm_angle = angle_diff / max_angle_rad
                if norm_dist + norm_angle > 1.0:
                    continue

                heapq.heappush(candidates, (dist, nbr))

            if not candidates:
                break

            _, next_idx = heapq.heappop(candidates)
            visited.add(next_idx)
            segment.appendleft(next_idx)
            current = next_idx

            # Update backward direction from the new front of the segment
            if len(segment) >= 2:
                bwd_angle = compute_edge_angle_fast(
                    points[segment[1], 0],
                    points[segment[1], 1],
                    points[segment[0], 0],
                    points[segment[0], 1],
                )

    return list(segment)

def clean_ends(segments):
    cleaned = []
    for seg_idx, segment in enumerate(segments):
        length = len(segment)
        if length < 5:
            cleaned.append(segment)
            continue


        # Compute turning-angle curvature at each interior point
        coords = np.array([(p[0], p[1]) for p in segment])
        vectors = np.diff(coords, axis=0)
        lengths = np.linalg.norm(vectors, axis=1)

        total_len = np.sum(lengths)
        len_threshold = 0.2 * total_len

        cumulative_len = np.cumsum(lengths)
        first_index = np.where(cumulative_len > len_threshold)[0][0]
        last_index = np.where(cumulative_len > (total_len - len_threshold))[0][0]

        angles = np.arctan2(vectors[:, 1], vectors[:, 0])
        curvatures = np.abs(np.diff(angles))
        curvatures = np.minimum(curvatures, 2 * np.pi - curvatures)
        avg_lengths = (lengths[:-1] + lengths[1:]) / 2.0
        curvatures = np.where(avg_lengths > 0, curvatures / avg_lengths, 0.0)

        # Average curvature of the middle portion (excluding the ends under review)
        if first_index > last_index:
            stdev_curvature = np.std(curvatures)
            average_curvature = np.mean(curvatures)
        else:
            stdev_curvature = np.std(curvatures[first_index:last_index])
            average_curvature = np.mean(curvatures[first_index:last_index])

        threshold = average_curvature
        
        print(f"Segment {seg_idx}: threshold = {np.degrees(threshold):.2f}, std = {np.degrees(stdev_curvature):.2f}, total length = {total_len:.2f} ft")
        print(f"first_index = {first_index}, last_index = {last_index}")

        # Trim consecutive high-curvature points from the start
        trim_start = 0
        for i in range(min(first_index, len(curvatures))):
            if curvatures[i] > threshold:
                trim_start = i + 1
            else:
                break

        # Trim consecutive high-curvature points from the end
        trim_end = 0
        for i in range(min(last_index, len(curvatures))):
            idx = len(curvatures) - 1 - i
            if curvatures[idx] > threshold:
                trim_end = i + 1
            else:
                break

        end_idx = length - trim_end if trim_end > 0 else length
        trimmed = list(segment[trim_start:end_idx])

        # Extend start with a straight segment to make up removed length
        if trim_start > 0 and len(trimmed) >= 2:
            removed_len = cumulative_len[trim_start - 1]
            direction = coords[trim_start + 1] - coords[trim_start]
            norm = np.linalg.norm(direction)
            if norm > 0:
                new_start = coords[trim_start] - (direction / norm) * removed_len
                trimmed.insert(0, (new_start[0], new_start[1]))

        # Extend end with a straight segment to make up removed length
        if trim_end > 0 and len(trimmed) >= 2:
            removed_len = total_len - cumulative_len[end_idx - 2]
            direction = coords[end_idx - 1] - coords[end_idx - 2]
            norm = np.linalg.norm(direction)
            if norm > 0:
                new_end = coords[end_idx - 1] + (direction / norm) * removed_len
                trimmed.append((new_end[0], new_end[1]))

        cleaned.append(trimmed if len(trimmed) >= 2 else segment)

    return cleaned

def filter_outliers_and_connect_optimized(
    points, distance_threshold=5.0, angle_threshold=30.0, minimum_length=5.0
):
    """Optimized outlier filtering using deque and vectorized operations."""
    points = np.array(points)
    if len(points) < 2:
        return [], [(pt[0], pt[1], "Not enough points") for pt in points]

    tree = KDTree(points)
    pairs = tree.query_pairs(r=distance_threshold)

    adjacency = [set() for _ in range(len(points))]
    edges = {}

    # Vectorize edge computation where possible
    for i, j in pairs:
        adjacency[i].add(j)
        adjacency[j].add(i)
        distance = np.linalg.norm(points[i] - points[j])
        angle = compute_edge_angle_fast(
            points[i, 0], points[i, 1], points[j, 0], points[j, 1]
        )
        edges[(i, j)] = {"angle": angle, "distance": distance}

    visited = set()
    polylines = []

    for i in range(len(points)):
        if i not in visited:
            segment_indices = bfs_connect_optimized(
                i,
                adjacency,
                edges,
                visited,
                points,
                angle_threshold,
                distance_threshold,
            )
            if len(segment_indices) >= 2:
                total_length = sum(
                    np.linalg.norm(
                        points[segment_indices[k]] - points[segment_indices[k - 1]]
                    )
                    for k in range(1, len(segment_indices))
                )
                if total_length >= minimum_length:
                    segment_points = [tuple(points[idx]) for idx in segment_indices]
                    polylines.append(segment_points)

    outliers = [
        (points[i][0], points[i][1], "Isolated point")
        for i in range(len(points))
        if i not in visited
    ]

    # Filter by average segment length
    filtered_polylines = []
    for seg in polylines:
        if len(seg) < 2:
            continue
        avg_len = np.mean(
            [
                np.linalg.norm(np.array(seg[k]) - np.array(seg[k - 1]))
                for k in range(1, len(seg))
            ]
        )
        if avg_len <= 1:
            filtered_polylines.append(seg)

    return filtered_polylines, outliers


def get_delaunay_centerlines(
    roadway_items,
    road_threshold=(5, 15.0),
    vertex_cluster_threshold=10,
    parallel_angle_epsilon=15.0,
):
    """
    Optimized Delaunay centerline extraction with:
    - Vectorized triangle filtering
    - Precomputed edge angles
    - Numba JIT for hot loops
    - Deque for O(1) BFS operations
    - Batch parallel checking
    """
    # 1. Collect and Resample all points
    all_points = []
    point_to_roadway_idx = []
    roadway_edges = {}
    min_road_width = road_threshold[0]
    max_road_width = road_threshold[1]

    for roadway_idx, item in enumerate(roadway_items):
        verts = item["verts"]
        if not verts or len(verts) < 2:
            continue

        roadway_edges[roadway_idx] = []
        for i in range(len(verts) - 1):
            roadway_edges[roadway_idx].append(
                (np.array(verts[i]), np.array(verts[i + 1]))
            )

        resampled = resample_polyline(verts, step_distance=1.0)
        for pt in resampled:
            all_points.append(pt)
            point_to_roadway_idx.append(roadway_idx)

    all_points = np.array(all_points)
    point_to_roadway_idx = np.array(point_to_roadway_idx)

    if len(all_points) < 3:
        return {
            "centerlines": [],
            "midpoints": [],
            "skeleton_segments": [],
            "debug": {
                "all_points": [],
                "triangle_segments": [],
                "clustered_triangles": [],
                "non_parallel_triangles": [],
                "undersized_triangles": [],
                "oversized_triangles": [],
                "vertex_cluster_info": {},
                "outlier_info": {},
            },
        }

    # 2. Precompute edge data structures
    edge_index, edge_list = build_edge_spatial_index(roadway_edges)
    edge_starts, edge_ends, edge_roadway_idx, roadway_edge_ranges = (
        build_roadway_edge_arrays(roadway_edges)
    )

    # 3. Compute Delaunay Triangulation
    tri = Delaunay(all_points)
    n_triangles = len(tri.simplices)

    # 4. Compute perpendicular road widths for all triangles
    perp_widths = compute_perpendicular_widths(
        tri.simplices, all_points, point_to_roadway_idx
    )

    # 5. Filter by perpendicular width
    perp_width_valid_mask = (perp_widths <= max_road_width) & (
        perp_widths >= min_road_width
    )

    # 6. VECTORIZED: Check roadway diversity per triangle
    tri_roadways = point_to_roadway_idx[tri.simplices]  # (n_tri, 3)
    # Count unique roadways per triangle
    roadway_diversity = np.array(
        [len(set(tri_roadways[i])) for i in range(n_triangles)]
    )
    diversity_valid_mask = roadway_diversity >= 2

    # 7. Compute vertex triangle counts for cluster detection
    vertex_triangle_count = {}
    for tri_idx, simplex in enumerate(tri.simplices):
        for vertex_idx in simplex:
            if vertex_idx not in vertex_triangle_count:
                vertex_triangle_count[vertex_idx] = []
            vertex_triangle_count[vertex_idx].append(tri_idx)

    problematic_vertices = set()
    vertex_cluster_info = {}
    for vertex_idx, triangle_list in vertex_triangle_count.items():
        count = len(triangle_list)
        vertex_cluster_info[vertex_idx] = count
        if count >= vertex_cluster_threshold:
            problematic_vertices.add(vertex_idx)

    clustered_triangle_indices = set()
    for vertex_idx in problematic_vertices:
        clustered_triangle_indices.update(vertex_triangle_count[vertex_idx])

    cluster_mask = np.array(
        [i not in clustered_triangle_indices for i in range(n_triangles)]
    )

    # 8. Combined first-pass filter
    valid_first_pass = perp_width_valid_mask & diversity_valid_mask & cluster_mask
    valid_indices = np.where(valid_first_pass)[0]

    # 9. BATCH: Check parallel edges for valid triangles
    is_parallel = check_parallel_vectorized(
        tri.simplices[valid_indices],
        all_points,
        point_to_roadway_idx,
        edge_starts,
        edge_ends,
        roadway_edge_ranges,
        parallel_angle_epsilon,
    )

    # 10. Build output structures
    skeleton_segments = []
    triangle_segments = []
    clustered_triangles = []
    non_parallel_triangles = []
    undersized_triangles = []
    oversized_triangles = []
    segments = set()
    skeleton_widths = []  # parallel to skeleton_segments

    # Process all triangles for visualization
    for i, simplex in enumerate(tri.simplices):
        pts = all_points[simplex]
        is_clustered = i in clustered_triangle_indices

        # Check if this triangle passed first-pass filter
        first_pass_idx = np.where(valid_indices == i)[0]
        is_par = False
        if len(first_pass_idx) > 0:
            is_par = is_parallel[first_pass_idx[0]]

        # Store triangle edges for visualization
        for j in range(3):
            start_pt = tuple(pts[j])
            end_pt = tuple(pts[(j + 1) % 3])
            if (start_pt, end_pt) not in segments and (
                end_pt,
                start_pt,
            ) not in segments:
                segments.add((start_pt, end_pt))
                if is_clustered:
                    clustered_triangles.append((start_pt, end_pt))
                elif not perp_width_valid_mask[i]:
                    if perp_widths[i] < min_road_width:
                        undersized_triangles.append((start_pt, end_pt))
                    else:
                        oversized_triangles.append((start_pt, end_pt))
                elif not diversity_valid_mask[i]:
                    continue  # Skip non-valid triangles
                elif not is_par:
                    non_parallel_triangles.append((start_pt, end_pt))
                else:
                    triangle_segments.append((start_pt, end_pt))

        # Skip centroid generation for invalid triangles
        if not valid_first_pass[i]:
            continue
        first_pass_idx = np.where(valid_indices == i)[0]
        if len(first_pass_idx) == 0 or not is_parallel[first_pass_idx[0]]:
            continue

        # Generate skeleton segments
        triangle_roadway_indices = set(point_to_roadway_idx[simplex])
        center_curr = np.mean(pts, axis=0)

        perp_w = perp_widths[i]

        for neighbor_idx in tri.neighbors[i]:
            if neighbor_idx == -1 or neighbor_idx < i:
                continue
            if neighbor_idx in clustered_triangle_indices:
                continue

            nbr_pts = all_points[tri.simplices[neighbor_idx]]
            if perp_widths[neighbor_idx] > max_road_width:
                continue

            center_nbr = np.mean(nbr_pts, axis=0)
            if np.linalg.norm(center_curr - center_nbr) > max_road_width / 2:
                continue

            neighbor_roadway_indices = set(
                point_to_roadway_idx[tri.simplices[neighbor_idx]]
            )
            relevant_roadways = triangle_roadway_indices | neighbor_roadway_indices

            if not check_crossing_optimized(
                center_curr, center_nbr, edge_index, edge_list, relevant_roadways
            ):
                skeleton_segments.append((center_curr, center_nbr))
                # Average perpendicular width of the two triangles
                nbr_w = perp_widths[neighbor_idx]
                seg_w = (perp_w + nbr_w) / 2.0
                skeleton_widths.append(seg_w)

    # 11. Compute midpoints, track widths, filter, and split wide roads
    midpoints = [
        ((seg[0][0] + seg[1][0]) / 2.0, (seg[0][1] + seg[1][1]) / 2.0)
        for seg in skeleton_segments
    ]

    # Build width lookup: midpoint coords -> perpendicular road width
    midpoint_width_map = {}
    for mp, w in zip(midpoints, skeleton_widths):
        midpoint_width_map[mp] = w

    centerline_segments = []
    outlier_info_passA = {}

    if midpoints:
        centerline_segments, outlier_info_passA = filter_outliers_and_connect_optimized(
            midpoints, distance_threshold=10, angle_threshold=15
        )

    # Smooth segments before splitting to avoid amplifying noise in offsets

    # Split wide roads into offset centerlines
    centerline_segments = split_wide_roads(
        centerline_segments,
        midpoint_width_map,
        split_threshold=22,
        triple_split_threshold=32.0,
    )

    centerline_segments = smooth_segments(centerline_segments, window=3)
    
    # Clean up ends of segments
    # centerline_segments = clean_ends(centerline_segments)

    # Second pass: reconnect split centerlines at endpoints with clothoid bridges
    # centerline_segments, outlier_info_passB = connect_endpoints_clothoid(
    #     centerline_segments, distance_threshold=50, angle_threshold=5   )

    all_centerline_points = [pt for seg in centerline_segments for pt in seg]

    # Build width-annotated segments from the (possibly split) centerlines
    segments_width = []
    for seg in centerline_segments:
        seg_widths = []
        for pt in seg:
            w = midpoint_width_map.get(pt, max_road_width)
            seg_widths.append((pt[0], pt[1], w))
        segments_width.append(seg_widths)

    return {
        "centerlines": segments_width,
        "midpoints": midpoints,
        "skeleton_segments": skeleton_segments,
        "debug": {
            "all_points": all_points,
            "triangle_segments": triangle_segments,
            "clustered_triangles": clustered_triangles,
            "non_parallel_triangles": non_parallel_triangles,
            "undersized_triangles": undersized_triangles,
            "oversized_triangles": oversized_triangles,
            "vertex_cluster_info": vertex_cluster_info,
            "all_centerline_points": all_centerline_points,
            # "outlier_info": {"passA": outlier_info_passA, "passB": outlier_info_passB},
        },
    }

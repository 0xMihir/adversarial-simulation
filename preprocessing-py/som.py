"""
Set-of-Mark (SoM) prompting for lane connection identification at intersections.

This module annotates lane endpoints with numbered markers, sends the annotated
diagram to a VLM, and parses the response to identify valid lane connections.
"""

import io
import base64
import json
import re
from dataclasses import dataclass
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from scipy.spatial import KDTree
from openai import OpenAI

try:
    from shapely.geometry import LineString, Polygon as ShapelyPolygon
    from shapely.prepared import prep
    HAS_SHAPELY = True
except ImportError:
    HAS_SHAPELY = False


@dataclass
class LaneEndpoint:
    """Represents an endpoint of a lane centerline."""
    id: int
    x: float
    y: float
    heading: float  # radians, direction the lane points INTO the intersection
    lane_idx: int   # index of the parent lane in centerlines list
    # is_start: bool  # True if this is the start of the lane, False if end


@dataclass
class LaneConnection:
    """Represents a connection between two lane endpoints."""
    from_id: int
    to_id: int
    confidence: float = 1.0


def bezier_curve(
    p0: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
    p3: np.ndarray,
    n_points: int = 50,
) -> np.ndarray:
    """
    Generate points along a cubic Bezier curve.

    Args:
        p0, p1, p2, p3: Control points (2D arrays)
        n_points: Number of points to sample

    Returns:
        Array of shape (n_points, 2) with curve points
    """
    t = np.linspace(0, 1, n_points).reshape(-1, 1)
    curve = (
        (1 - t) ** 3 * p0
        + 3 * (1 - t) ** 2 * t * p1
        + 3 * (1 - t) * t ** 2 * p2
        + t ** 3 * p3
    )
    return curve


def bezier_from_endpoints(
    from_pos: np.ndarray,
    from_heading: float,
    to_pos: np.ndarray,
    to_heading: float,
    control_distance_factor: float = 0.4,
    n_points: int = 50,
) -> np.ndarray:
    """
    Generate a Bezier curve connecting two endpoints with specified headings.

    Control points are placed along the heading directions.

    Args:
        from_pos: Starting position (x, y)
        from_heading: Heading at start (radians, pointing INTO intersection)
        to_pos: Ending position (x, y)
        to_heading: Heading at end (radians, pointing INTO intersection)
        control_distance_factor: Fraction of endpoint distance for control points
        n_points: Number of curve points

    Returns:
        Array of shape (n_points, 2) with curve points
    """
    p0 = np.array(from_pos)
    p3 = np.array(to_pos)

    # Distance between endpoints
    dist = np.linalg.norm(p3 - p0)
    control_dist = dist * control_distance_factor

    # P1: from start, along reversed heading (away from intersection center)
    # The heading points INTO the intersection, so we reverse it for outgoing direction
    p1 = p0 + control_dist * np.array([
        -np.cos(from_heading),
        -np.sin(from_heading)
    ])

    # P2: from end, along heading direction (into the intersection from end's perspective)
    p2 = p3 + control_dist * np.array([
        np.cos(to_heading),
        np.sin(to_heading)
    ])

    return bezier_curve(p0, p1, p2, p3, n_points)


def extract_closed_polygons(roadway: list[dict]) -> list[np.ndarray]:
    """
    Extract closed polygons from roadway items (typically curbs, islands).

    Args:
        roadway: List of roadway items from scene["roadway"]

    Returns:
        List of polygon vertex arrays
    """
    closed_polys = []
    for item in roadway:
        if item.get("closed", False):
            verts = np.array(item["verts"])
            closed_polys.append(verts)
    return closed_polys


def check_curve_collision(
    curve_points: np.ndarray,
    closed_polygons: list[np.ndarray],
) -> tuple[bool, list[int]]:
    """
    Check if a curve intersects any closed polygons.

    Args:
        curve_points: Array of shape (n, 2) with curve points
        closed_polygons: List of polygon vertex arrays

    Returns:
        (has_collision, colliding_polygon_indices)
    """
    if not HAS_SHAPELY:
        # Fallback: no collision detection without shapely
        return False, []

    if len(curve_points) < 2:
        return False, []

    curve_line = LineString(curve_points)
    colliding_indices = []

    for i, poly_verts in enumerate(closed_polygons):
        if len(poly_verts) < 3:
            continue
        try:
            poly = ShapelyPolygon(poly_verts)
            if not poly.is_valid:
                poly = poly.buffer(0)  # Fix invalid polygons
            if curve_line.intersects(poly):
                colliding_indices.append(i)
        except Exception:
            continue

    return len(colliding_indices) > 0, colliding_indices


def generate_obstacle_free_curve(
    from_ep: "LaneEndpoint",
    to_ep: "LaneEndpoint",
    closed_polygons: list[np.ndarray],
    n_points: int = 50,
    max_iterations: int = 10,
    control_factor_range: tuple[float, float] = (0.2, 0.8),
) -> tuple[np.ndarray, bool]:
    """
    Generate a Bezier curve that avoids closed polygons.

    Iteratively adjusts control point distances to find a collision-free path.

    Args:
        from_ep: Starting endpoint
        to_ep: Ending endpoint
        closed_polygons: List of obstacle polygons
        n_points: Number of curve points
        max_iterations: Maximum adjustment iterations
        control_factor_range: Range of control distance factors to try

    Returns:
        (curve_points, is_collision_free)
    """
    from_pos = np.array([from_ep.x, from_ep.y])
    to_pos = np.array([to_ep.x, to_ep.y])

    # Try different control factors
    factors_to_try = np.linspace(
        control_factor_range[0],
        control_factor_range[1],
        max_iterations
    )

    best_curve = None
    best_collision_count = float('inf')

    for factor in factors_to_try:
        curve = bezier_from_endpoints(
            from_pos, from_ep.heading,
            to_pos, to_ep.heading,
            control_distance_factor=factor,
            n_points=n_points,
        )

        has_collision, colliding_indices = check_curve_collision(
            curve, closed_polygons
        )

        if not has_collision:
            return curve, True

        if len(colliding_indices) < best_collision_count:
            best_collision_count = len(colliding_indices)
            best_curve = curve

    # Return best attempt even if it has collisions
    return best_curve if best_curve is not None else bezier_from_endpoints(
        from_pos, from_ep.heading,
        to_pos, to_ep.heading,
        n_points=n_points,
    ), False


def compute_heading(points: list[tuple], at_start: bool) -> float:
    """
    Compute the heading at the start or end of a lane.
    Returns angle in radians pointing INTO the intersection.
    """
    if len(points) < 2:
        return 0.0

    if at_start:
        # At start: heading points from point[1] to point[0] (into intersection)
        p0 = np.array(points[0][:2])
        p1 = np.array(points[1][:2])
        direction = p0 - p1
    else:
        # At end: heading points from point[-2] to point[-1] (into intersection)
        p0 = np.array(points[-1][:2])
        p1 = np.array(points[-2][:2])
        direction = p0 - p1

    return np.arctan2(direction[1], direction[0])


def extract_lane_endpoints(
    centerlines: list[list[tuple]],
    cluster_radius: float = 20.0,
) -> tuple[list[LaneEndpoint], list[int]]:
    """
    Extract endpoints from centerlines that are near an intersection.

    An intersection is detected as a region where multiple lane endpoints cluster.

    Args:
        centerlines: List of lanes, each lane is [(x, y, width), ...]
        cluster_radius: Distance threshold for clustering endpoints

    Returns:
        endpoints: List of LaneEndpoint objects at intersection zones
        intersection_endpoint_ids: List of endpoint IDs that are at intersections
    """
    # Collect all endpoints
    all_endpoints = []
    endpoint_id = 0

    for lane_idx, lane in enumerate(centerlines):
        if len(lane) < 2:
            continue

        # End endpoint
        start_pt = lane[0]
        start_heading = compute_heading(lane, at_start=True)
        all_endpoints.append(LaneEndpoint(
            id=endpoint_id,
            x=start_pt[0],
            y=start_pt[1],
            heading=start_heading,
            lane_idx=lane_idx,
        ))
        endpoint_id += 1

        # Start endpoint
        end_pt = lane[-1]
        end_heading = compute_heading(lane, at_start=False)
        all_endpoints.append(LaneEndpoint(
            id=endpoint_id,
            x=end_pt[0],
            y=end_pt[1],
            heading=end_heading,
            lane_idx=lane_idx,
        ))
        endpoint_id += 1

    print(f"Extracted {len(all_endpoints)} total endpoints from centerlines")
    print(all_endpoints[0])
    if not all_endpoints:
        return [], []

    # Build KDTree to find clusters
    coords = np.array([[ep.x, ep.y] for ep in all_endpoints])
    tree = KDTree(coords)

    # Find endpoints that have neighbors (intersection candidates)
    intersection_ids = set()
    for i, ep in enumerate(all_endpoints):
        neighbors = tree.query_ball_point([ep.x, ep.y], cluster_radius)
        # Intersection if there are other endpoints nearby from different lanes
        neighbor_lanes = {all_endpoints[j].lane_idx for j in neighbors}
        if len(neighbor_lanes) > 1:
            intersection_ids.update(neighbors)

    return all_endpoints, list(intersection_ids)


def create_annotated_figure(
    roadway: list[dict],
    centerlines: list[list[tuple]],
    endpoints: list[LaneEndpoint],
    intersection_ids: list[int],
    figsize: tuple[int, int] = (12, 10),
    marker_size: int = 200,
    font_size: int = 10,
) -> plt.Figure:
    """
    Create a matplotlib figure with numbered markers at lane endpoints.

    Args:
        roadway: List of roadway items from scene["roadway"]
        centerlines: List of lane centerlines [(x, y, width), ...]
        endpoints: List of all LaneEndpoint objects
        intersection_ids: List of endpoint IDs to mark (intersection endpoints)
        figsize: Figure size
        marker_size: Size of the numbered markers
        font_size: Font size for marker labels

    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Draw roadway
    for item in roadway:
        verts = item["verts"]
        xs, ys = zip(*verts)
        dashed = item.get("dashed", False)
        closed = item.get("closed", False)
        thick = item.get("thick", False)

        if dashed:
            ax.plot(xs, ys, color='black', linestyle='--', linewidth=2 if thick else 1)
        else:
            ax.plot(xs, ys, color='black', linewidth=2 if thick else 1)
            if closed:
                ax.fill(xs, ys, color='lightgray', alpha=0.5)

    # Draw centerlines
    segments = []
    for lane in centerlines:
        for i in range(len(lane) - 1):
            x0, y0, _ = lane[i]
            x1, y1, _ = lane[i + 1]
            segments.append([[x0, y0], [x1, y1]])

    if segments:
        lc = LineCollection(segments, color='blue', linewidths=2, alpha=0.9, zorder=5)
        ax.add_collection(lc)

    # Draw numbered markers at intersection endpoints
    # intersection_endpoints = [ep for ep in endpoints if ep.id in intersection_ids]

    for ep in endpoints:
        ax.scatter(ep.x, ep.y, s=marker_size, c="red", marker='o',
                   zorder=10)

        # Draw number label
        ax.annotate(
            str(ep.id),
            (ep.x, ep.y),
            fontsize=font_size,
            fontweight='bold',
            color='white',
            ha='center',
            va='center',
            zorder=11,
        )

        # # Draw heading arrow
        # arrow_len = 15.0
        # dx = arrow_len * np.cos(ep.heading)
        # dy = arrow_len * np.sin(ep.heading)
        # ax.annotate(
        #     '',
        #     xy=(ep.x + dx, ep.y + dy),
        #     xytext=(ep.x, ep.y),
        #     arrowprops=dict(arrowstyle='->', color='yellow', lw=2),
        #     zorder=9,
        # )

    ax.set_aspect('equal', 'box')
    ax.axis('off')

    return fig


def figure_to_base64(fig: plt.Figure, format: str = 'png', dpi: int = 150) -> str:
    """Convert a matplotlib figure to a base64-encoded string."""
    buf = io.BytesIO()
    fig.savefig(buf, format=format, dpi=dpi, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


def build_som_prompt(
    endpoints: list[LaneEndpoint],
    intersection_ids: list[int],
) -> str:
    """
    Build the SoM prompt for the VLM.

    Args:
        endpoints: List of all LaneEndpoint objects
        intersection_ids: List of endpoint IDs that are marked in the image

    Returns:
        Prompt string for the VLM
    """
    intersection_endpoints = [ep for ep in endpoints if ep.id in intersection_ids]

    # Build endpoint descriptions
    endpoint_info = []
    for ep in intersection_endpoints:
        heading_deg = np.degrees(ep.heading) % 360
        endpoint_info.append(f"  - Marker {ep.id}: Lane {ep.lane_idx}, heading {heading_deg:.0f}°")

    endpoint_list = "\n".join(endpoint_info)

    prompt = f"""This image shows a road intersection diagram with lane centerlines (blue lines) and numbered markers (red circles with white numbers) at lane endpoints.

The yellow arrows show the direction each lane points INTO the intersection.

Markers in this image:
{endpoint_list}

Task: Identify which lane endpoints can legally connect to form valid paths through the intersection.

Rules for valid connections:
1. The connection path should follow realistic traffic flow (no sharp U-turns within the intersection)
2. Consider lane positions - leftmost entering lanes typically connect to leftmost exiting lanes
3. Each lane can connect to multiple valid exits (e.g., left turn, straight, right turn)

Respond with a JSON array of connections. Each connection should have:
- "from": the marker ID of the entering lane
- "to": the marker ID of the exiting lane
- "type": one of "left_turn", "straight", "right_turn", "u_turn"

Example response format:
```json
[
  {{"from": 0, "to": 3, "type": "straight"}},
  {{"from": 0, "to": 5, "type": "left_turn"}},
  {{"from": 2, "to": 7, "type": "right_turn"}}
]
```

Analyze the intersection and provide the connections:"""

    return prompt


def query_vlm(
    image_base64: str,
    prompt: str,
    api_key: str,
    base_url: str = "https://api.openai.com/v1",
    model: str = "gpt-4o",
    max_tokens: int = 2048,
) -> str:
    """
    Query the VLM with the annotated image and prompt.

    Args:
        image_base64: Base64-encoded image
        prompt: The SoM prompt
        api_key: API key for the service
        base_url: Base URL for the API (use for llama-server compatibility)
        model: Model name to use
        max_tokens: Maximum response tokens

    Returns:
        Raw response text from the VLM
    """
    client = OpenAI(api_key=api_key, base_url=base_url)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_base64}",
                        },
                    },
                    {
                        "type": "text",
                        "text": prompt,
                    },
                ],
            }
        ],
        max_tokens=max_tokens,
    )
    
    print(response)

    return response.choices[0].message.content


def parse_vlm_response(response: str) -> list[LaneConnection]:
    """
    Parse the VLM response to extract lane connections.

    Args:
        response: Raw response text from VLM

    Returns:
        List of LaneConnection objects
    """
    # Try to extract JSON from the response
    # Look for JSON array pattern
    json_match = re.search(r'\[[\s\S]*?\]', response)

    if not json_match:
        raise ValueError(f"Could not find JSON array in response: {response}")

    try:
        connections_data = json.loads(json_match.group())
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON: {e}\nRaw: {json_match.group()}")

    connections = []
    for item in connections_data:
        connections.append(LaneConnection(
            from_id=int(item["from"]),
            to_id=int(item["to"]),
            confidence=float(item.get("confidence", 1.0)),
        ))

    return connections


def validate_connections(
    connections: list[LaneConnection],
    endpoints: list[LaneEndpoint],
    intersection_ids: list[int],
) -> list[LaneConnection]:
    """
    Validate and filter connections to ensure they are valid.

    Args:
        connections: List of proposed connections from VLM
        endpoints: List of all LaneEndpoint objects
        intersection_ids: List of valid endpoint IDs

    Returns:
        List of validated LaneConnection objects
    """
    ep_lookup = {ep.id: ep for ep in endpoints}
    valid_ids = set(intersection_ids)

    validated = []
    for conn in connections:
        # Check IDs are valid
        if conn.from_id not in valid_ids or conn.to_id not in valid_ids:
            continue

        from_ep = ep_lookup.get(conn.from_id)
        to_ep = ep_lookup.get(conn.to_id)

        if not from_ep or not to_ep:
            continue

        # Check we're not connecting a lane to itself
        if from_ep.lane_idx == to_ep.lane_idx:
            continue

        # Check direction: from should be entering (is_start=True), to should be exiting (is_start=False)
        # OR vice versa depending on lane orientation
        # For now, just ensure they're different lanes
        validated.append(conn)

    return validated


def identify_lane_connections(
    roadway: list[dict],
    centerlines: list[list[tuple]],
    api_key: str,
    base_url: str = "https://api.openai.com/v1",
    model: str = "gpt-4o",
    cluster_radius: float = 20.0,
    debug: bool = False,
) -> dict:
    """
    Main function to identify lane connections using SoM prompting.

    Args:
        roadway: List of roadway items from scene["roadway"]
        centerlines: List of lane centerlines from get_delaunay_centerlines()["centerlines"]
        api_key: API key for the VLM service
        base_url: Base URL for the API
        model: Model name to use
        cluster_radius: Distance threshold for detecting intersection zones
        debug: If True, returns additional debug info

    Returns:
        Dictionary with:
            - connections: List of LaneConnection objects
            - endpoints: List of all LaneEndpoint objects
            - intersection_ids: List of endpoint IDs at intersections
            - figure: The annotated matplotlib figure (if debug=True)
            - prompt: The prompt sent to VLM (if debug=True)
            - raw_response: Raw VLM response (if debug=True)
    """
    # Extract endpoints
    endpoints, intersection_ids = extract_lane_endpoints(centerlines, cluster_radius)

    if not intersection_ids:
        return {
            "connections": [],
            "endpoints": endpoints,
            "intersection_ids": [],
            "message": "No intersection detected (no clustered endpoints)",
        }

    # Create annotated figure
    fig = create_annotated_figure(
        roadway, centerlines, endpoints, intersection_ids
    )

    # Convert to base64
    image_b64 = figure_to_base64(fig)

    # Build prompt
    prompt = build_som_prompt(endpoints, intersection_ids)

    # Query VLM
    raw_response = query_vlm(
        image_b64, prompt, api_key, base_url, model, max_tokens=16384
    )
    print("VLM Response:", raw_response)

    # Parse response
    connections = parse_vlm_response(raw_response)

    # Validate connections
    validated = validate_connections(connections, endpoints, intersection_ids)

    result = {
        "connections": validated,
        "endpoints": endpoints,
        "intersection_ids": intersection_ids,
    }

    if debug:
        result["figure"] = fig
        result["prompt"] = prompt
        result["raw_response"] = raw_response
        result["raw_connections"] = connections
    else:
        plt.close(fig)

    return result


def draw_connections(
    ax: plt.Axes,
    connections: list[LaneConnection],
    endpoints: list[LaneEndpoint],
    roadway: list[dict] = None,
    color: str = 'green',
    linewidth: float = 3,
    alpha: float = 0.7,
    use_bezier: bool = True,
    avoid_obstacles: bool = True,
) -> dict:
    """
    Draw the identified connections on an existing axes.

    Args:
        ax: Matplotlib axes to draw on
        connections: List of LaneConnection objects
        endpoints: List of LaneEndpoint objects
        roadway: List of roadway items (needed for obstacle avoidance)
        color: Color for connection lines
        linewidth: Width of connection lines
        alpha: Transparency
        use_bezier: If True, use Bezier curves; if False, use straight lines
        avoid_obstacles: If True, attempt to avoid closed polygons (curbs)

    Returns:
        Dictionary with curve data for each connection
    """
    ep_lookup = {ep.id: ep for ep in endpoints}
    curve_data = {}

    # Extract closed polygons for obstacle avoidance
    closed_polygons = []
    if roadway and avoid_obstacles:
        closed_polygons = extract_closed_polygons(roadway)

    for conn in connections:
        from_ep = ep_lookup.get(conn.from_id)
        to_ep = ep_lookup.get(conn.to_id)

        if not from_ep or not to_ep:
            continue

        if use_bezier:
            # Generate Bezier curve with optional obstacle avoidance
            if avoid_obstacles and closed_polygons:
                curve_points, collision_free = generate_obstacle_free_curve(
                    from_ep, to_ep, closed_polygons
                )
            else:
                curve_points = bezier_from_endpoints(
                    np.array([from_ep.x, from_ep.y]), from_ep.heading,
                    np.array([to_ep.x, to_ep.y]), to_ep.heading,
                )
                collision_free = True

            # Store curve data
            curve_data[(conn.from_id, conn.to_id)] = {
                'points': curve_points,
                'collision_free': collision_free,
            }

            # Draw the curve
            ax.plot(
                curve_points[:, 0],
                curve_points[:, 1],
                color=color if collision_free else 'orange',
                linewidth=linewidth,
                alpha=alpha,
                zorder=8,
            )

            # Draw arrow at midpoint
            mid_idx = len(curve_points) // 2
            if mid_idx > 0 and mid_idx < len(curve_points) - 1:
                ax.annotate(
                    '',
                    xy=(curve_points[mid_idx + 1, 0], curve_points[mid_idx + 1, 1]),
                    xytext=(curve_points[mid_idx - 1, 0], curve_points[mid_idx - 1, 1]),
                    arrowprops=dict(arrowstyle='->', color=color, lw=linewidth),
                    zorder=9,
                )
        else:
            # Straight line fallback
            ax.plot(
                [from_ep.x, to_ep.x],
                [from_ep.y, to_ep.y],
                color=color,
                linewidth=linewidth,
                alpha=alpha,
                zorder=8,
                linestyle='--',
            )

            # Draw arrow at midpoint
            mid_x = (from_ep.x + to_ep.x) / 2
            mid_y = (from_ep.y + to_ep.y) / 2
            dx = to_ep.x - from_ep.x
            dy = to_ep.y - from_ep.y

            ax.annotate(
                '',
                xy=(mid_x + dx * 0.1, mid_y + dy * 0.1),
                xytext=(mid_x - dx * 0.1, mid_y - dy * 0.1),
                arrowprops=dict(arrowstyle='->', color=color, lw=linewidth),
                zorder=8,
            )

    return curve_data

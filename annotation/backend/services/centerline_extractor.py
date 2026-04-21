"""
Wraps preprocessing/delaunay.py and maps output to list[LaneAnnotation].
All lanes start with status=AUTO.
"""
import uuid

from faro import get_delaunay_centerlines
from schema.annotation import ElementStatus, LaneAnnotation
from schema.scene import ParsedScene, Point2D


def _roadway_items_from_scene(scene: ParsedScene) -> list[dict]:
    """Reconstruct the roadway item dicts that delaunay.py expects."""
    items = []
    for idx in scene.roadway_indices:
        elem = scene.elements[idx]
        # delaunay.py expects dicts with 'transformed_verts' key
        verts = [(p.x, p.y) for p in elem.resampled_points]
        items.append({
            "type": elem.element_type,
            "transformed_verts": verts,
            "dashed": elem.is_dashed,
            "closed": elem.is_closed,
        })
    return items


def extract_centerlines(
    scene: ParsedScene,
    road_threshold: tuple[float, float] = (8, 36.0),
    vertex_cluster_threshold: int = 10,
    parallel_angle_epsilon: float = 15.0,
) -> list[LaneAnnotation]:
    """
    Run Delaunay-based centerline extraction on the parsed scene's roadway elements.

    Returns a list of LaneAnnotation objects with status=AUTO.
    """
    roadway_items = _roadway_items_from_scene(scene)
    if not roadway_items:
        return []

    result = get_delaunay_centerlines(
        roadway_items,
        road_threshold=road_threshold,
        vertex_cluster_threshold=vertex_cluster_threshold,
        parallel_angle_epsilon=parallel_angle_epsilon,
    )

    lanes: list[LaneAnnotation] = []
    for i, centerline in enumerate(result.get("centerlines", [])):
        # centerline is [(x, y, width), ...]
        polyline = [Point2D(x=float(pt[0]), y=float(pt[1])) for pt in centerline]
        if len(polyline) < 2:
            continue

        lane = LaneAnnotation(
            id=f"L{i + 1:03d}",
            polyline=polyline,
            raw_control_points=polyline,  # centerlines are already the output geometry
            lane_type="driving",
            left_boundary_type="unknown",
            right_boundary_type="unknown",
            entry_lanes=[],
            exit_lanes=[],
            status=ElementStatus.AUTO,
        )
        lanes.append(lane)

    return lanes

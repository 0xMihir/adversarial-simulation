"""
Wraps preprocessing/traj.py and maps output to list[VehicleAnnotation].
All vehicles start with status=AUTO.
"""
from faro import extract_vehicle_chronology
from schema.annotation import ElementStatus, VehicleAnnotation, VehicleWaypoint
from schema.scene import ParsedScene, Point2D


def _scene_for_traj(scene: ParsedScene) -> dict:
    """
    Reconstruct the scene dict that traj.py's extract_vehicle_chronology() expects.
    traj.py expects scene["vehicles"] as a list of symbol dicts with:
      - transformed_center: (x, y)
      - items: [...] containing label items with text
      - transform: 3x3 numpy matrix
    """
    import numpy as np

    vehicles = []
    for veh in scene.vehicles:
        mat_vals = None
        if veh.source_element_ids:
            # Find first source element to get transform
            for elem in scene.elements:
                if elem.id in veh.source_element_ids:
                    mat_vals = elem.transform.values
                    break

        mat = np.array(mat_vals) if mat_vals else np.eye(3)

        # Build label items from associated text
        label_items = []
        if veh.label_text:
            label_items.append({
                "type": "label",
                "text": veh.label_text,
                "transformed_center": (veh.center.x, veh.center.y),
            })

        # Build symbol-like dict
        vehicles.append({
            "type": "symbol",
            "name": veh.label_text,
            "transformed_center": (veh.center.x, veh.center.y),
            "center": (veh.center.x, veh.center.y),
            "bbox": (
                min(p.x for p in veh.obb) if veh.obb else veh.center.x - 5,
                min(p.y for p in veh.obb) if veh.obb else veh.center.y - 2,
                max(p.x for p in veh.obb) if veh.obb else veh.center.x + 5,
                max(p.y for p in veh.obb) if veh.obb else veh.center.y + 2,
            ),
            "transform": mat,
            "items": label_items,
            "associated_text": [veh.label_text] if veh.label_text else [],
            "vehicle2d": True,
            "dashed": False,
        })

    return {"vehicles": vehicles}


def _phase_label(phase_enum) -> str:
    """Map traj.py MotionPhase to annotation phase string."""
    from faro import MotionPhase
    mapping = {
        MotionPhase.NORMAL_DRIVING: "pre_crash",
        MotionPhase.LOSS_OF_CONTROL: "collision",
        MotionPhase.POST_COLLISION: "post_crash",
    }
    return mapping.get(phase_enum, "pre_crash")


def fit_trajectories(scene: ParsedScene) -> list[VehicleAnnotation]:
    """
    Run TSP + clothoid trajectory fitting on parsed vehicle detections.

    Returns a list of VehicleAnnotation with status=AUTO.
    """
    traj_scene = _scene_for_traj(scene)

    if not traj_scene["vehicles"]:
        return []

    try:
        result = extract_vehicle_chronology(traj_scene, pre_crash_only=False)
    except Exception as e:
        # Trajectory fitting can fail on degenerate cases; return minimal annotations
        return _fallback_from_scene(scene)

    vehicle_annotations: list[VehicleAnnotation] = []

    fitted = result.get("fitted_trajectories", {})
    trajectories = result.get("trajectories", {})

    for label, segments in fitted.items():
        waypoints: list[VehicleWaypoint] = []
        ts_idx = 0
        for seg in segments:
            phase_str = _phase_label(seg.phase)
            for pos in seg.positions:
                if isinstance(pos, dict):
                    tc = pos.get("transformed_center") or pos.get("center")
                    if not tc or len(tc) < 2:
                        continue
                    x, y = float(tc[0]), float(tc[1])
                    t = pos.get("transform")
                    if t is not None:
                        import numpy as np
                        sx = float(np.sqrt(t[0, 0]**2 + t[1, 0]**2)) or 1.0
                        heading = float(np.arctan2(t[1, 0] / sx, t[0, 0] / sx))
                    else:
                        heading = 0.0
                elif hasattr(pos, '__len__') and not isinstance(pos, dict) and len(pos) >= 2:
                    x, y = float(pos[0]), float(pos[1])
                    heading = float(pos[2]) if len(pos) > 2 else 0.0
                else:
                    continue
                waypoints.append(VehicleWaypoint(
                    position=Point2D(x=x, y=y),
                    heading=heading,
                    timestamp_index=ts_idx,
                    phase=phase_str,
                ))
                ts_idx += 1

        if not waypoints and label in trajectories:
            # Fall back to raw ordered positions
            for ts_idx, pos in enumerate(trajectories[label]):
                if hasattr(pos, 'center'):
                    tc = pos.transformed_center
                    x, y = float(tc[0]), float(tc[1])
                else:
                    continue
                waypoints.append(VehicleWaypoint(
                    position=Point2D(x=x, y=y),
                    heading=0.0,
                    timestamp_index=ts_idx,
                    phase="pre_crash",
                ))

        if waypoints:
            vehicle_annotations.append(VehicleAnnotation(
                id=str(label),
                vehicle_type="car",
                waypoints=waypoints,
                status=ElementStatus.AUTO,
            ))

    # Add vehicles with no trajectory data
    annotated_ids = {va.id for va in vehicle_annotations}
    for veh in scene.vehicles:
        if veh.id not in annotated_ids:
            vehicle_annotations.append(VehicleAnnotation(
                id=veh.id,
                vehicle_type="car",
                waypoints=[VehicleWaypoint(
                    position=veh.center,
                    heading=veh.heading,
                    timestamp_index=0,
                    phase="pre_crash",
                )],
                status=ElementStatus.AUTO,
            ))

    return vehicle_annotations


def _fallback_from_scene(scene: ParsedScene) -> list[VehicleAnnotation]:
    """Return minimal vehicle annotations directly from detected OBBs."""
    annotations = []
    for veh in scene.vehicles:
        annotations.append(VehicleAnnotation(
            id=veh.id,
            vehicle_type="car",
            waypoints=[VehicleWaypoint(
                position=veh.center,
                heading=veh.heading,
                timestamp_index=0,
                phase="pre_crash",
            )],
            status=ElementStatus.AUTO,
        ))
    return annotations

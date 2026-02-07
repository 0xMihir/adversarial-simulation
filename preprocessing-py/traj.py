# ============================================================================
# VEHICLE TRAJECTORY ORDERING AND COLLISION DETECTION
# ============================================================================

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
from enum import Enum
from itertools import permutations
from scipy.interpolate import CubicSpline, interp1d
from pyclothoids import Clothoid


class MotionPhase(Enum):
    """Classification of vehicle motion phase."""
    NORMAL_DRIVING = "normal"      # Nonholonomic, clothoid-valid
    LOSS_OF_CONTROL = "loc"        # High slip angle, transitional
    POST_COLLISION = "post"        # Free body motion, spinning/sliding


@dataclass
class SlipMetrics:
    """Metrics for detecting loss-of-control conditions."""
    slip_angle: float          # Angle between heading and velocity (radians)
    yaw_rate: float            # Heading change rate (rad/m)
    path_curvature: float      # Curvature implied by position change
    heading_curvature: float   # Curvature implied by heading change
    curvature_mismatch: float  # |path_curvature - heading_curvature|


@dataclass
class CrashTrajectorySegment:
    """A segment of trajectory with phase information."""
    start_idx: int
    end_idx: int
    phase: MotionPhase
    positions: List[Dict]
    path: Optional[np.ndarray] = None  # Sampled (x, y, theta, t) points

class OrientedBoundingBox:
    """OBB for higher-fidelity collision detection."""
    
    def __init__(self, center, half_extents, rotation_angle):
        self.center = np.array(center)  # (x, y) in WORLD coordinates
        self.half_extents = np.array(half_extents)  # scaled dimensions
        self.angle = rotation_angle  # radians

    @classmethod
    def from_vehicle_symbol(cls, symbol):
        """Extract OBB from vehicle symbol using transform matrix."""
        transform = symbol["transform"]

        # Extract scale factors from the transform
        # For M = T @ R @ S: column vectors have length = scale
        sx = np.sqrt(transform[0, 0]**2 + transform[1, 0]**2)
        sy = np.sqrt(transform[0, 1]**2 + transform[1, 1]**2)

        # Extract rotation (after removing scale)
        angle = np.arctan2(transform[1, 0] / sx, transform[0, 0] / sx)

        # Get LOCAL bbox and compute half extents, then APPLY SCALE
        bbox = symbol["bbox"]
        half_w = (bbox[2] - bbox[0]) / 2 * sx
        half_h = (bbox[3] - bbox[1]) / 2 * sy

        center = symbol["transformed_center"]
        return cls(center, (half_w, half_h), angle)

    def get_corners(self):
        """Return 4 corners of the OBB in world coordinates."""
        c, s = np.cos(self.angle), np.sin(self.angle)
        R = np.array([[c, -s], [s, c]])

        corners_local = np.array([
            [-self.half_extents[0], -self.half_extents[1]],
            [ self.half_extents[0], -self.half_extents[1]],
            [ self.half_extents[0],  self.half_extents[1]],
            [-self.half_extents[0],  self.half_extents[1]],
        ])
        return (R @ corners_local.T).T + self.center


def obb_intersect(obb1, obb2):
    """Separating Axis Theorem (SAT) for OBB-OBB intersection."""
    def get_axes(obb):
        c, s = np.cos(obb.angle), np.sin(obb.angle)
        return [np.array([c, s]), np.array([-s, c])]

    corners1 = obb1.get_corners()
    corners2 = obb2.get_corners()

    for axis in get_axes(obb1) + get_axes(obb2):
        proj1 = corners1 @ axis
        proj2 = corners2 @ axis
        if max(proj1) < min(proj2) or max(proj2) < min(proj1):
            return False  # Separating axis found
    return True  # No separating axis = collision


def project_obb_onto_axis(obb, axis):
    """Project OBB half-extent onto given axis."""
    c, s = np.cos(obb.angle), np.sin(obb.angle)
    local_axes = [np.array([c, s]), np.array([-s, c])]

    return sum(
        abs(np.dot(local_axis, axis)) * half_ext
        for local_axis, half_ext in zip(local_axes, obb.half_extents)
    )


def compute_penetration_depth(obb1, obb2):
    """
    Compute approximate penetration depth between two OBBs.
    Simplified: distance between centers minus sum of extents along center axis.
    """
    center_diff = obb2.center - obb1.center
    center_dist = np.linalg.norm(center_diff)

    if center_dist < 1e-6:
        # OBBs are concentric, return max extent
        return np.linalg.norm(obb1.half_extents) + np.linalg.norm(obb2.half_extents)

    # Project extents onto center-to-center axis
    axis = center_diff / center_dist
    proj1 = project_obb_onto_axis(obb1, axis)
    proj2 = project_obb_onto_axis(obb2, axis)

    # Penetration = overlap of projections
    return max(0, (proj1 + proj2) - center_dist)


# ============================================================================
# VEHICLE LABEL VALIDATION AND GROUPING
# ============================================================================

def validate_vehicle_labels(vehicles):
    """Ensure all vehicles have labels. Raise error if any unlabeled."""
    unlabeled = [v for v in vehicles if not v["associated_text"]
                 or not any(t.strip().isdigit() for t in v["associated_text"])]
    if unlabeled:
        raise ValueError(f"Found {len(unlabeled)} unlabeled vehicles. "
                        "User annotation required.")
    return True


def group_vehicles_by_label(vehicles):
    """Group all vehicle positions by their label number."""
    groups = {}  # label -> list of vehicle symbols
    for v in vehicles:
        label = next((t for t in v["associated_text"] if t.strip().isdigit()), None)
        if label:
            groups.setdefault(label, []).append(v)
    return groups


# ============================================================================
# TSP-BASED TRAJECTORY ORDERING (Distance + Angle Score)
# ============================================================================

def get_vehicle_heading(symbol):
    """Extract heading angle from vehicle transform matrix."""
    transform = symbol["transform"]
    sx = np.sqrt(transform[0, 0]**2 + transform[1, 0]**2)
    return np.arctan2(transform[1, 0] / sx, transform[0, 0] / sx)


def angle_difference(a1, a2):
    """Compute smallest signed angle difference."""
    diff = a1 - a2
    return np.arctan2(np.sin(diff), np.cos(diff))


# ============================================================================
# SLIP METRICS AND MOTION PHASE CLASSIFICATION
# ============================================================================

def compute_slip_metrics(pos_from, pos_to):
    """
    Compute metrics that indicate whether vehicle is in normal driving
    or loss-of-control state.
    """
    from_pos = np.array(pos_from["transformed_center"])
    to_pos = np.array(pos_to["transformed_center"])
    from_heading = get_vehicle_heading(pos_from)
    to_heading = get_vehicle_heading(pos_to)

    delta_pos = to_pos - from_pos
    distance = np.linalg.norm(delta_pos)

    if distance < 1e-6:
        return SlipMetrics(0, 0, 0, 0, 0)

    velocity_angle = np.arctan2(delta_pos[1], delta_pos[0])
    slip_angle = angle_difference(from_heading, velocity_angle)

    delta_heading = angle_difference(to_heading, from_heading)
    yaw_rate = delta_heading / distance

    path_curvature = delta_heading / distance
    heading_curvature = delta_heading / distance
    curvature_mismatch = abs(path_curvature - heading_curvature)

    return SlipMetrics(
        slip_angle=slip_angle,
        yaw_rate=yaw_rate,
        path_curvature=path_curvature,
        heading_curvature=heading_curvature,
        curvature_mismatch=curvature_mismatch,
    )


def classify_motion_phase(
    pos_from, pos_to,
    slip_threshold=np.pi / 6,       # 30 degrees
    yaw_rate_threshold=0.5,          # rad/m
    collision_detected=False,
):
    """
    Classify the motion between two waypoints.

    Returns:
        MotionPhase classification
    """
    if collision_detected:
        return MotionPhase.POST_COLLISION

    metrics = compute_slip_metrics(pos_from, pos_to)

    if abs(metrics.slip_angle) > slip_threshold:
        return MotionPhase.LOSS_OF_CONTROL

    if abs(metrics.yaw_rate) > yaw_rate_threshold:
        return MotionPhase.LOSS_OF_CONTROL

    return MotionPhase.NORMAL_DRIVING


# ============================================================================
# PHASE-AWARE EDGE SCORING
# ============================================================================

def _freebody_score(from_pos, to_pos, from_heading, to_heading):
    """
    Score for free-body (sliding/spinning) motion.

    Position and heading evolve independently (unlike clothoid where they're
    coupled). Score based on energy: linear sliding + angular spinning can
    happen simultaneously.
    """
    distance = np.linalg.norm(np.array(to_pos) - np.array(from_pos))
    heading_change = abs(angle_difference(to_heading, from_heading))

    linear_score = distance
    angular_score = heading_change * 2.0  # meters-equivalent

    return np.sqrt(linear_score**2 + angular_score**2)


def _clothoid_arc_length(from_pos, to_pos, from_heading, to_heading):
    """Compute clothoid arc length between two poses."""
    try:
        clothoid = Clothoid.G1Hermite(
            from_pos[0], from_pos[1], from_heading,
            to_pos[0], to_pos[1], to_heading,
        )
        return abs(clothoid.Parameters[-1])
    except Exception:
        return np.linalg.norm(np.array(to_pos) - np.array(from_pos)) * 10


def compute_edge_score_phase_aware(pos_from, pos_to, phase=None):
    """
    Compute edge score using the appropriate model for the motion phase.

    - NORMAL_DRIVING: clothoid arc length (nonholonomic)
    - LOSS_OF_CONTROL: min of clothoid and freebody (transitional)
    - POST_COLLISION: freebody score (holonomic, heading != velocity)

    Returns:
        (score, detected_phase)
    """
    if phase is None:
        phase = classify_motion_phase(pos_from, pos_to)

    from_pos = np.array(pos_from["transformed_center"])
    to_pos = np.array(pos_to["transformed_center"])
    from_heading = get_vehicle_heading(pos_from)
    to_heading = get_vehicle_heading(pos_to)

    if phase == MotionPhase.NORMAL_DRIVING:
        score = _clothoid_arc_length(from_pos, to_pos, from_heading, to_heading)

    elif phase == MotionPhase.LOSS_OF_CONTROL:
        clothoid_score = _clothoid_arc_length(from_pos, to_pos, from_heading, to_heading)
        freebody = _freebody_score(from_pos, to_pos, from_heading, to_heading)
        score = min(clothoid_score, freebody * 1.5)  # slight LOC penalty

    else:  # POST_COLLISION
        score = _freebody_score(from_pos, to_pos, from_heading, to_heading)

    return score, phase


# ============================================================================
# EDGE SCORING (ORIGINAL — kept as fallback)
# ============================================================================

def compute_edge_score(pos_from, pos_to, distance_weight=1.0, angle_weight=0.5):
    """
    Score for transitioning from pos_from to pos_to.
    Lower score = more physically plausible transition.
    """
    from_pos = np.array(pos_from["transformed_center"])
    to_pos = np.array(pos_to["transformed_center"])

    # Distance component
    distance = np.linalg.norm(to_pos - from_pos)

    # Movement direction
    move_direction = np.arctan2(to_pos[1] - from_pos[1], to_pos[0] - from_pos[0])

    # Vehicle heading at "from" position
    from_heading = get_vehicle_heading(pos_from)

    # Angle between movement and heading (vehicle should move forward)
    heading_alignment = abs(angle_difference(move_direction, from_heading))
    angle_penalty = heading_alignment / np.pi  # 0 = forward, 1 = backward

    # Heading continuity (smooth rotation)
    to_heading = get_vehicle_heading(pos_to)
    heading_change = abs(angle_difference(to_heading, from_heading))
    continuity_penalty = heading_change / np.pi

    return (distance_weight * distance +
            angle_weight * angle_penalty * distance +
            angle_weight * 0.5 * continuity_penalty * distance)


def compute_path_score(positions, order):
    """Compute total score for a given ordering of positions."""
    total = 0.0
    ordered_positions = [positions[i] for i in order]
    for i in range(len(ordered_positions) - 1):
        total += compute_edge_score(ordered_positions[i], ordered_positions[i + 1])
    return total


def tsp_bruteforce(positions):
    """
    Find optimal ordering via brute-force permutation search.
    Feasible for n <= 8 (8! = 40320).
    """
    n = len(positions)
    if n <= 1:
        return list(range(n)), 0.0

    best_order = None
    best_score = float('inf')

    for perm in permutations(range(n)):
        score = compute_path_score(positions, perm)
        if score < best_score:
            best_score = score
            best_order = list(perm)

    return best_order, best_score


def tsp_greedy_2opt(positions):
    """
    Greedy nearest-neighbor with 2-opt improvement.
    For n > 8 where brute-force is too slow.
    """
    n = len(positions)
    if n <= 1:
        return list(range(n)), 0.0

    # Greedy nearest-neighbor starting from each position
    best_order = None
    best_score = float('inf')

    for start in range(n):
        order = [start]
        remaining = set(range(n)) - {start}

        while remaining:
            current = order[-1]
            # Find nearest neighbor by edge score
            best_next = min(remaining,
                           key=lambda x: compute_edge_score(positions[current], positions[x]))
            order.append(best_next)
            remaining.remove(best_next)

        # 2-opt improvement
        improved = True
        while improved:
            improved = False
            for i in range(1, n - 1):
                for j in range(i + 1, n):
                    # Try reversing segment [i:j+1]
                    new_order = order[:i] + order[i:j+1][::-1] + order[j+1:]
                    new_score = compute_path_score(positions, new_order)
                    if new_score < compute_path_score(positions, order):
                        order = new_order
                        improved = True

        score = compute_path_score(positions, order)
        if score < best_score:
            best_score = score
            best_order = order

    return best_order, best_score


def order_vehicle_trajectory_tsp(positions, bruteforce_threshold=8):
    """
    Order vehicle positions using TSP with the score function.

    Args:
        positions: list of vehicle position symbols
        bruteforce_threshold: use brute-force for n <= this value

    Returns:
        ordered: list of positions in chronological order
        score: total path score (lower = better)
    """
    n = len(positions)
    if n <= 1:
        return positions, 0.0

    if n <= bruteforce_threshold:
        order, score = tsp_bruteforce(positions)
    else:
        order, score = tsp_greedy_2opt(positions)

    return [positions[i] for i in order], score


def connect_trajectory(ordered_positions, num_samples=100):
    """Connect ordered poses with clothoid curves, return sampled (x, y) lists."""
    if len(ordered_positions) < 2:
        return [], []

    all_x, all_y = [], []
    for i in range(len(ordered_positions) - 1):
        p0 = ordered_positions[i]
        p1 = ordered_positions[i + 1]
        c0 = p0["transformed_center"]
        c1 = p1["transformed_center"]
        h0 = get_vehicle_heading(p0)
        h1 = get_vehicle_heading(p1)
        try:
            clothoid = Clothoid.G1Hermite(c0[0], c0[1], h0, c1[0], c1[1], h1)
            xs, ys = clothoid.SampleXY(num_samples)
            all_x.extend(xs)
            all_y.extend(ys)
        except Exception:
            all_x.extend([c0[0], c1[0]])
            all_y.extend([c0[1], c1[1]])
    return all_x, all_y


# ============================================================================
# CRASH-AWARE TSP ORDERING
# ============================================================================

def compute_path_score_crash_aware(positions, order, collision_position_set=None):
    """
    Compute total path score with phase-aware edge scoring.

    Args:
        positions: list of vehicle position symbols
        order: proposed ordering (list of indices)
        collision_position_set: set of position indices known to be collision points
    """
    if collision_position_set is None:
        collision_position_set = set()

    total = 0.0
    ordered_positions = [positions[i] for i in order]

    for i in range(len(ordered_positions) - 1):
        is_collision = (order[i] in collision_position_set
                        or order[i + 1] in collision_position_set)
        phase = MotionPhase.POST_COLLISION if is_collision else None
        score, _ = compute_edge_score_phase_aware(
            ordered_positions[i], ordered_positions[i + 1], phase=phase
        )
        total += score

    return total


def order_vehicle_trajectory_crash_aware(
    positions, collision_indices=None, bruteforce_threshold=8
):
    """
    Order vehicle positions using crash-aware TSP scoring.

    Args:
        positions: list of vehicle position symbols
        collision_indices: list of position indices known to be collision points
        bruteforce_threshold: use brute-force for n <= this value

    Returns:
        (ordered_positions, score)
    """
    n = len(positions)
    if n <= 1:
        return positions, 0.0

    collision_set = set(collision_indices) if collision_indices else set()

    def score_func(order):
        return compute_path_score_crash_aware(positions, order, collision_set)

    if n <= bruteforce_threshold:
        best_order = None
        best_score = float('inf')
        for perm in permutations(range(n)):
            score = score_func(list(perm))
            if score < best_score:
                best_score = score
                best_order = list(perm)
    else:
        # Greedy nearest-neighbor + 2-opt using phase-aware scoring
        best_order = None
        best_score = float('inf')

        for start in range(n):
            order = [start]
            remaining = set(range(n)) - {start}

            while remaining:
                current = order[-1]
                best_next = min(
                    remaining,
                    key=lambda x: compute_edge_score_phase_aware(
                        positions[current], positions[x]
                    )[0],
                )
                order.append(best_next)
                remaining.remove(best_next)

            # 2-opt improvement
            improved = True
            while improved:
                improved = False
                current_score = score_func(order)
                for i in range(1, n - 1):
                    for j in range(i + 1, n):
                        new_order = order[:i] + order[i:j+1][::-1] + order[j+1:]
                        new_score = score_func(new_order)
                        if new_score < current_score:
                            order = new_order
                            current_score = new_score
                            improved = True

            score = score_func(order)
            if score < best_score:
                best_score = score
                best_order = order

    return [positions[i] for i in best_order], best_score


# ============================================================================
# POST-COLLISION TRAJECTORY FITTING
# ============================================================================

def fit_post_collision_trajectory(positions, collision_idx, dt=0.1):
    """
    Fit a physically plausible post-collision trajectory using spline
    interpolation. Position and heading are interpolated independently
    (holonomic — no nonholonomic constraint).

    Args:
        positions: ordered waypoints for one vehicle
        collision_idx: index where collision occurred (relative to positions)
        dt: time step for output trajectory

    Returns:
        Array of shape (N, 4) with columns (x, y, theta, t)
    """
    post = positions[collision_idx:]
    if len(post) < 2:
        return np.array([])

    xy = np.array([p["transformed_center"] for p in post])
    headings = np.unwrap([get_vehicle_heading(p) for p in post])

    # Estimate time parameterization from inter-waypoint distances
    friction_decel = 7.0  # m/s^2
    times = [0.0]
    for i in range(1, len(xy)):
        dist = np.linalg.norm(xy[i] - xy[i - 1])
        speed = max(5.0, 20.0 - friction_decel * times[-1])
        times.append(times[-1] + dist / speed)
    times = np.array(times)

    # Fit splines — cubic for 4+ points, linear otherwise
    if len(post) >= 4:
        sx = CubicSpline(times, xy[:, 0], bc_type="natural")
        sy = CubicSpline(times, xy[:, 1], bc_type="natural")
        st = CubicSpline(times, headings, bc_type="natural")
    else:
        sx = interp1d(times, xy[:, 0], kind="linear", fill_value="extrapolate")
        sy = interp1d(times, xy[:, 1], kind="linear", fill_value="extrapolate")
        st = interp1d(times, headings, kind="linear", fill_value="extrapolate")

    t_samples = np.arange(0, times[-1], dt)
    trajectory = np.column_stack([sx(t_samples), sy(t_samples),
                                  st(t_samples), t_samples])
    return trajectory


def _fit_clothoid_path(positions, dt=0.1):
    """Fit clothoid path for a normal-driving segment."""
    if len(positions) < 2:
        return np.array([])

    all_points = []
    for i in range(len(positions) - 1):
        c0 = positions[i]["transformed_center"]
        c1 = positions[i + 1]["transformed_center"]
        h0 = get_vehicle_heading(positions[i])
        h1 = get_vehicle_heading(positions[i + 1])

        try:
            clothoid = Clothoid.G1Hermite(c0[0], c0[1], h0, c1[0], c1[1], h1)
            arc_length = abs(clothoid.Parameters[-1])
            n_samples = max(2, int(arc_length / dt))
            for j in range(n_samples):
                s = j * arc_length / n_samples
                all_points.append([clothoid.X(s), clothoid.Y(s),
                                   clothoid.Theta(s), 0])
        except Exception:
            all_points.append([c0[0], c0[1], h0, 0])

    last = positions[-1]
    all_points.append([last["transformed_center"][0],
                       last["transformed_center"][1],
                       get_vehicle_heading(last), 0])
    return np.array(all_points)


def _detect_collision_index(positions):
    """
    Detect likely collision point from trajectory dynamics.
    Returns the index with the highest slip/yaw anomaly.
    """
    max_anomaly = 0
    collision_idx = 0

    for i in range(len(positions) - 1):
        metrics = compute_slip_metrics(positions[i], positions[i + 1])
        anomaly = abs(metrics.slip_angle) + abs(metrics.yaw_rate)
        if anomaly > max_anomaly:
            max_anomaly = anomaly
            collision_idx = i

    return collision_idx


def fit_hybrid_trajectory(positions, collision_idx=None):
    """
    Fit a hybrid trajectory: clothoid pre-collision, spline post-collision.

    Args:
        positions: ordered waypoints for one vehicle
        collision_idx: index where collision occurred (auto-detect if None)

    Returns:
        list of CrashTrajectorySegment with fitted paths
    """
    if collision_idx is None:
        collision_idx = _detect_collision_index(positions)

    segments = []

    # Pre-collision: normal driving (clothoid)
    if collision_idx > 0:
        pre = positions[:collision_idx + 1]
        seg = CrashTrajectorySegment(
            start_idx=0, end_idx=collision_idx,
            phase=MotionPhase.NORMAL_DRIVING, positions=pre,
        )
        seg.path = _fit_clothoid_path(pre)
        segments.append(seg)

    # Post-collision: free-body (spline)
    if collision_idx < len(positions) - 1:
        post = positions[collision_idx:]
        seg = CrashTrajectorySegment(
            start_idx=collision_idx, end_idx=len(positions) - 1,
            phase=MotionPhase.POST_COLLISION, positions=post,
        )
        seg.path = fit_post_collision_trajectory(post, collision_idx=0)
        segments.append(seg)

    return segments


# ============================================================================
# COLLISION DETECTION (EXHAUSTIVE FOR SMALL N)
# ============================================================================

def find_all_collision_pairs(vehicle_trajectories):
    """
    Find all position pairs that result in collision between any two vehicles.

    Args:
        vehicle_trajectories: dict of label -> ordered list of positions

    Returns:
        list of collision info dicts
    """
    labels = list(vehicle_trajectories.keys())
    collisions = []

    # Build OBBs for all positions
    all_obbs = {}
    for label, positions in vehicle_trajectories.items():
        all_obbs[label] = [OrientedBoundingBox.from_vehicle_symbol(p) for p in positions]

    # Check ALL position pairs between vehicles (exhaustive for small N)
    for i, label1 in enumerate(labels):
        for label2 in labels[i+1:]:
            for idx1, obb1 in enumerate(all_obbs[label1]):
                for idx2, obb2 in enumerate(all_obbs[label2]):
                    if obb_intersect(obb1, obb2):
                        collisions.append({
                            "vehicles": (label1, label2),
                            "indices": (idx1, idx2),
                            "obb1": obb1,
                            "obb2": obb2,
                            "penetration": compute_penetration_depth(obb1, obb2),
                        })

    return collisions


def find_first_collision(vehicle_trajectories):
    """
    Find each vehicle's FIRST collision in chronological order.

    Strategy:
    1. Find all colliding position pairs
    2. For each vehicle, find its earliest collision by normalized trajectory
       index (earlier index = earlier in time)
    3. Break ties by minimum penetration (closest to first contact)
    4. Deduplicate: if two vehicles share the same collision event, store it
       once and reference it from both

    Returns:
        dict with:
        - per_vehicle: dict of label -> collision info dict (the vehicle's
          first collision). Two labels may map to the same collision object
          when their first collisions are the same event.
        - unique_collisions: list of distinct first-collision events

        Returns None if no collisions found.
    """
    collisions = find_all_collision_pairs(vehicle_trajectories)

    if not collisions:
        return None

    def normalized_index(collision, label):
        """Normalized trajectory index for a vehicle in this collision."""
        label1, label2 = collision["vehicles"]
        idx = collision["indices"][0] if label == label1 else collision["indices"][1]
        n = len(vehicle_trajectories[label])
        return idx / max(1, n - 1)

    # For each vehicle, find its earliest collision
    per_vehicle = {}
    for collision in collisions:
        label1, label2 = collision["vehicles"]
        for label in (label1, label2):
            norm_idx = normalized_index(collision, label)
            key = (norm_idx, collision["penetration"])
            if label not in per_vehicle:
                per_vehicle[label] = (key, collision)
            else:
                if key < per_vehicle[label][0]:
                    per_vehicle[label] = (key, collision)

    # Unwrap to just collision dicts
    per_vehicle = {label: col for label, (_, col) in per_vehicle.items()}

    # Deduplicate: collect unique collision objects (by identity)
    seen_ids = set()
    unique_collisions = []
    for col in per_vehicle.values():
        if id(col) not in seen_ids:
            seen_ids.add(id(col))
            unique_collisions.append(col)

    return {
        "per_vehicle": per_vehicle,
        "unique_collisions": unique_collisions,
    }


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def extract_vehicle_chronology(scene, pre_crash_only=False):
    """
    Extract chronological vehicle ordering with crash-aware trajectory fitting.

    Two-phase model:
    - Pre-collision: nonholonomic (clothoid) — vehicle is driving normally
    - Post-collision: holonomic dynamics — position and heading evolve
      independently (sliding, spinning)

    Args:
        scene: Parsed scene from FaroSceneGraphReader
        pre_crash_only: If True, only compute nonholonomic (clothoid)
            trajectories up to the first collision point for each vehicle.
            Skips crash-aware re-ordering, slip classification, and
            post-collision spline fitting.

    Returns:
        dict with:
        - trajectories: dict of label -> ordered positions
        - connected: dict of label -> (x_list, y_list) clothoid-connected path
        - fitted_trajectories: dict of label -> list of CrashTrajectorySegment
        - first_collisions: per-vehicle first collision info or None
        - collision_indices: dict of label -> list of collision position indices
        - scores: dict of label -> path score (lower = better)
        - all_collisions: list of all collision pairs

    Raises:
        ValueError: If any vehicles are unlabeled (user must annotate)
    """
    vehicles = scene["vehicles"]

    # Step 1: Validate labels
    validate_vehicle_labels(vehicles)

    # Step 2: Group by label
    groups = group_vehicles_by_label(vehicles)

    # Step 3: Initial ordering using original TSP
    trajectories = {}
    scores = {}
    for label, group in groups.items():
        ordered, score = order_vehicle_trajectory_tsp(group)
        trajectories[label] = ordered
        scores[label] = score

    # Step 4: Detect collisions
    all_collisions = find_all_collision_pairs(trajectories)
    first_collisions = find_first_collision(trajectories)

    # Step 5: Build collision index map per vehicle
    collision_indices = {}
    for col in all_collisions:
        label1, label2 = col["vehicles"]
        idx1, idx2 = col["indices"]
        collision_indices.setdefault(label1, []).append(idx1)
        collision_indices.setdefault(label2, []).append(idx2)

    if pre_crash_only:
        # Truncate trajectories at first collision and fit clothoid only
        truncated_trajectories = {}
        connected = {}
        fitted_trajectories = {}
        for label, positions in trajectories.items():
            col_idxs = collision_indices.get(label, [])
            cut = min(col_idxs) + 1 if col_idxs else len(positions)
            truncated = positions[:cut]
            truncated_trajectories[label] = truncated
            connected[label] = connect_trajectory(truncated)
            seg = CrashTrajectorySegment(
                start_idx=0, end_idx=len(truncated) - 1,
                phase=MotionPhase.NORMAL_DRIVING, positions=truncated,
            )
            seg.path = _fit_clothoid_path(truncated)
            fitted_trajectories[label] = [seg]

        return {
            "trajectories": truncated_trajectories,
            "connected": connected,
            "fitted_trajectories": fitted_trajectories,
            "first_collisions": first_collisions,
            "collision_indices": collision_indices,
            "scores": scores,
            "all_collisions": all_collisions,
        }

    # Step 6: Re-order using crash-aware scoring where collisions exist
    for label, group in groups.items():
        col_idxs = collision_indices.get(label, [])
        if col_idxs:
            ordered, score = order_vehicle_trajectory_crash_aware(
                group, collision_indices=col_idxs,
            )
            trajectories[label] = ordered
            scores[label] = score

    # Step 7: Re-detect collisions with updated ordering
    all_collisions = find_all_collision_pairs(trajectories)
    first_collisions = find_first_collision(trajectories)

    # Rebuild collision indices after re-ordering
    collision_indices = {}
    for col in all_collisions:
        label1, label2 = col["vehicles"]
        idx1, idx2 = col["indices"]
        collision_indices.setdefault(label1, []).append(idx1)
        collision_indices.setdefault(label2, []).append(idx2)

    # Step 8: Connect trajectories with clothoid curves (for backwards compat)
    connected = {}
    for label, positions in trajectories.items():
        connected[label] = connect_trajectory(positions)

    # Step 9: Fit hybrid trajectories (clothoid pre-collision, spline post)
    fitted_trajectories = {}
    for label, positions in trajectories.items():
        col_idxs = collision_indices.get(label, [])
        col_idx = min(col_idxs) if col_idxs else None
        fitted_trajectories[label] = fit_hybrid_trajectory(positions, col_idx)

    return {
        "trajectories": trajectories,
        "connected": connected,
        "fitted_trajectories": fitted_trajectories,
        "first_collisions": first_collisions,
        "collision_indices": collision_indices,
        "scores": scores,
        "all_collisions": all_collisions,
    }
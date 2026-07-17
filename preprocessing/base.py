"""
Dialect-neutral geometry helpers shared by the .far (FARO) and .blz (ARAS
Blitz) scene readers, plus the constructors that define the primitive/symbol
dict contract consumed by annotation.backend.services.faro_parser and
preprocessing.traj.
"""
import numpy as np


def make_affine(tx, ty, sx, sy, rot):
    """3x3 affine from translate/scale/rotate(radians), composed T @ R @ S."""
    S = np.array([[sx, 0, 0], [0, sy, 0], [0, 0, 1]])
    c, s = np.cos(rot), np.sin(rot)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    T = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]])
    return T @ R @ S


def apply_transform(points, matrix):
    """Apply a 3x3 matrix to an (N,2) array (or list) of 2D points -> (N,2)."""
    pts = np.asarray(points, dtype=float)
    if pts.ndim == 1:
        pts = pts.reshape(1, -1)
    if len(pts) == 0:
        return np.empty((0, 2))
    ones = np.ones((pts.shape[0], 1))
    pts_homo = np.hstack([pts, ones])
    transformed = (matrix @ pts_homo.T).T
    return transformed[:, :2]


def avg_curvature_deg(verts: np.ndarray) -> float:
    """Average curvature in degrees per unit length."""
    if len(verts) < 3:
        return 0.0
    diffs = np.diff(verts, axis=0)
    seg_lens = np.linalg.norm(diffs, axis=1)
    headings = np.arctan2(diffs[:, 1], diffs[:, 0])
    dtheta = np.abs(np.diff(headings))
    dtheta = np.minimum(dtheta, 2 * np.pi - dtheta)
    avg_lens = (seg_lens[:-1] + seg_lens[1:]) / 2.0
    curv = np.where(avg_lens > 0, dtheta / avg_lens, 0.0)
    return float(np.degrees(np.mean(curv)))


def bezier_cubic(P0, C1, C2, P1, t):
    """Evaluate a cubic Bezier at parameters t (m,) -> (m,2)."""
    t = t[:, None]
    return (
        ((1 - t) ** 3) * P0
        + 3 * ((1 - t) ** 2) * t * C1
        + 3 * (1 - t) * (t**2) * C2
        + (t**3) * P1
    )


def bezier_quadratic(P0, C, P1, t):
    """Evaluate a quadratic Bezier at parameters t (m,) -> (m,2)."""
    t = t[:, None]
    return ((1 - t) ** 2) * P0 + 2 * (1 - t) * t * C + (t**2) * P1


def interp_bezier_composite(pts_2d, tension):
    """Composite cubic Bézier with Hobby-like tangent estimation."""
    N_SAMPLES = 10
    n = len(pts_2d)

    # Estimate tangents at each point
    tangents = np.zeros_like(pts_2d)
    for i in range(n):
        if i == 0:
            tangents[i] = pts_2d[1] - pts_2d[0]
        elif i == n - 1:
            tangents[i] = pts_2d[-1] - pts_2d[-2]
        else:
            tangents[i] = pts_2d[i + 1] - pts_2d[i - 1]

    all_pts = []
    for i in range(n - 1):
        p0 = pts_2d[i]
        p3 = pts_2d[i + 1]
        seg_len = np.linalg.norm(p3 - p0)
        p1 = p0 + tension * tangents[i] * seg_len / np.linalg.norm(tangents[i] + 1e-12)
        p2 = p3 - tension * tangents[i + 1] * seg_len / np.linalg.norm(tangents[i + 1] + 1e-12)

        t_seg = np.linspace(0, 1, max(N_SAMPLES // (n - 1), 20))
        curve = (
            np.outer((1 - t_seg) ** 3, p0)
            + np.outer(3 * (1 - t_seg) ** 2 * t_seg, p1)
            + np.outer(3 * (1 - t_seg) * t_seg**2, p2)
            + np.outer(t_seg**3, p3)
        )
        all_pts.append(curve if i == 0 else curve[1:])  # avoid duplicating junction points

    return np.vstack(all_pts)


def fit_arc_through_3pts(p_start, p_mid, p_end, n_samples=24) -> np.ndarray:
    """
    Sample a circular arc from p_start to p_end passing through p_mid
    (circumcircle fit). Falls back to a straight segment for collinear points.
    """
    p1 = np.asarray(p_start, dtype=float)
    p2 = np.asarray(p_mid, dtype=float)
    p3 = np.asarray(p_end, dtype=float)

    d = 2 * (p1[0] * (p2[1] - p3[1]) + p2[0] * (p3[1] - p1[1]) + p3[0] * (p1[1] - p2[1]))
    if abs(d) < 1e-9:
        return np.vstack([p1, p3])

    s1, s2, s3 = (p1 @ p1), (p2 @ p2), (p3 @ p3)
    cx = (s1 * (p2[1] - p3[1]) + s2 * (p3[1] - p1[1]) + s3 * (p1[1] - p2[1])) / d
    cy = (s1 * (p3[0] - p2[0]) + s2 * (p1[0] - p3[0]) + s3 * (p2[0] - p1[0])) / d
    center = np.array([cx, cy])
    radius = np.linalg.norm(p1 - center)

    a1 = np.arctan2(p1[1] - cy, p1[0] - cx)
    a2 = np.arctan2(p2[1] - cy, p2[0] - cx)
    a3 = np.arctan2(p3[1] - cy, p3[0] - cx)

    # Sweep from a1 to a3 in the direction that passes through a2
    ccw_mid = (a2 - a1) % (2 * np.pi)
    ccw_end = (a3 - a1) % (2 * np.pi)
    if ccw_mid <= ccw_end:
        sweep = ccw_end  # counter-clockwise
    else:
        sweep = ccw_end - 2 * np.pi  # clockwise

    angles = a1 + np.linspace(0.0, sweep, n_samples)
    return center + radius * np.column_stack([np.cos(angles), np.sin(angles)])


def sample_ellipse(cx, cy, rx, ry, rot=0.0, n=32) -> np.ndarray:
    """Sample a closed ellipse outline -> (n+1, 2), first point repeated last."""
    t = np.linspace(0, 2 * np.pi, n + 1)
    pts = np.column_stack([rx * np.cos(t), ry * np.sin(t)])
    if rot:
        c, s = np.cos(rot), np.sin(rot)
        pts = pts @ np.array([[c, s], [-s, c]])
    return pts + np.array([cx, cy])


_EMPTY2 = np.empty((0, 2))


def make_primitive_dict(
    el_type,
    name,
    verts,
    transform,
    *,
    control_points=None,
    bezier_handles=None,
    interpolation_method="passthrough",
    layer=None,
    oriz=0.0,
    text=None,
    dashed=False,
    thick=False,
    closed=False,
    vehicle2d=False,
    lndata=None,
) -> dict:
    """
    Build the flat primitive dict that downstream consumers expect. `verts`
    are in the primitive's local frame; world-frame copies are derived with
    `transform`.
    """
    verts = np.asarray(verts, dtype=float)
    control_points = _EMPTY2 if control_points is None else np.asarray(control_points, dtype=float)
    bezier_handles = _EMPTY2 if bezier_handles is None else np.asarray(bezier_handles, dtype=float)
    center = verts.mean(axis=0)
    bbox = (verts[:, 0].min(), verts[:, 1].min(), verts[:, 0].max(), verts[:, 1].max())
    return {
        "type": el_type,
        "name": name,
        "verts": verts,
        "transformed_verts": apply_transform(verts, transform),
        "control_points": control_points,
        "bezier_handles": bezier_handles,
        "transformed_control_points": apply_transform(control_points, transform),
        "transformed_bezier_handles": apply_transform(bezier_handles, transform),
        "interpolation_method": interpolation_method,
        "center": center,
        "transformed_center": apply_transform(center[np.newaxis], transform)[0],
        "bbox": bbox,
        "vehicle2d": vehicle2d,
        "transform": transform,
        "oriz": oriz,
        "text": text,
        "dashed": dashed,
        "thick": thick,
        "closed": closed,
        "layer": layer,
        "lndata": lndata if lndata is not None else {},
    }


def make_image_dict(name, cx, cy, sizx, sizy, oriz, img_data, layer, transform) -> dict:
    """Build the image/sign primitive dict (center + size, no stroke geometry)."""
    return {
        "type": "image",
        "name": name,
        "posx": cx,
        "posy": cy,
        "sizx": sizx,
        "sizy": sizy,
        "oriz": oriz,
        "img": img_data,
        "layer": layer,
        "verts": None,
        "transformed_verts": _EMPTY2,
        "control_points": _EMPTY2,
        "bezier_handles": _EMPTY2,
        "transformed_control_points": _EMPTY2,
        "transformed_bezier_handles": _EMPTY2,
        "interpolation_method": "none",
        "center": np.array([cx, cy]),
        "transformed_center": np.array([cx, cy]),
        "bbox": (cx - sizx / 2, cy - sizy / 2, cx + sizx / 2, cy + sizy / 2),
        "vehicle2d": False,
        "transform": transform,
        "text": None,
        "dashed": False,
        "thick": False,
        "closed": False,
        "lndata": {},
    }


def make_symbol_dict(name, items, transform, layer=None, vehicle2d=False) -> dict:
    """
    Build a symbol (group) dict from already-extracted child items. bbox is the
    union of child bboxes (children's local frames — see reader traversal),
    center its midpoint; transformed_center maps center through `transform`.
    """
    bbox = (float("inf"), float("inf"), float("-inf"), float("-inf"))
    for item in items:
        ibox = item["bbox"]
        bbox = (
            min(bbox[0], ibox[0]),
            min(bbox[1], ibox[1]),
            max(bbox[2], ibox[2]),
            max(bbox[3], ibox[3]),
        )
    center = (np.mean([bbox[0], bbox[2]]), np.mean([bbox[1], bbox[3]]))
    is_dashed = any(item.get("dashed", False) for item in items)
    return {
        "type": "symbol",
        "name": name,
        "items": items,
        "bbox": bbox,
        "center": center,
        "transformed_center": apply_transform([center], transform)[0],
        "vehicle2d": vehicle2d,
        "transform": transform,
        "layer": layer,
        "predicted_class": None,
        "predicted_probability": None,
        "dashed": is_dashed,
    }

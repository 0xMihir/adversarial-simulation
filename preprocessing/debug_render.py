"""
Debug renderer for parsed CISS scenes.

Accepts either a ParsedScene (schema.scene) or the raw scene_objects dict
returned by a scene-graph reader's parse().

Two modes:
- "annotated": colored by semantic bucket, vehicle OBBs with heading arrows,
  optional text labels. For golden-blessing review and notebooks.
- "strokes": every geometry as a uniform black stroke on white, no text or
  markers. Input to the vendor-render raster comparison.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

BUCKET_COLORS = {
    "roadway": "#666666",
    "road_markings": "#3070b3",
    "misc": "#b8a24a",
    "ground_marks": "#e07b30",
    "annotations": "#b06fc9",
    "vehicle": "#c0392b",
}


def _iter_strokes(scene):
    """Yield (xy_world (N,2), bucket_name, is_dashed) for every drawable stroke."""
    if hasattr(scene, "elements"):  # ParsedScene
        buckets = {
            "roadway": set(scene.roadway_indices),
            "road_markings": set(scene.road_marking_indices),
            "misc": set(scene.other_indices),
        }
        for i, el in enumerate(scene.elements):
            xy = el.resampled_xy if len(el.resampled_xy) else el.control_xy
            if xy is None or len(xy) == 0:
                continue
            bucket = next((b for b, idxs in buckets.items() if i in idxs), "misc")
            yield np.asarray(xy, dtype=float), bucket, bool(el.is_dashed)
        for veh in scene.vehicles:
            if veh.obb:
                obb = np.array([[p.x, p.y] for p in veh.obb] + [[veh.obb[0].x, veh.obb[0].y]])
                yield obb, "vehicle", False
        return

    # Raw scene_objects dict
    def _prim_strokes(prim, bucket):
        if prim.get("type") == "symbol":
            for sub in prim.get("items", []):
                yield from _prim_strokes(sub, bucket)
            return
        tv = prim.get("transformed_verts")
        if tv is not None and len(tv):
            yield np.asarray(tv, dtype=float), bucket, bool(prim.get("dashed", False))

    for bucket in ("roadway", "road_markings", "misc", "ground_marks", "annotations"):
        for prim in scene.get(bucket, []) or []:
            yield from _prim_strokes(prim, bucket)
    for sym in scene.get("vehicles", []) or []:
        yield from _prim_strokes(sym, "vehicle")


def _iter_vehicles(scene):
    """Yield (center (2,), heading_rad or None) per vehicle."""
    if hasattr(scene, "elements"):
        for veh in scene.vehicles:
            yield np.array([veh.center.x, veh.center.y]), veh.heading
        return
    for sym in scene.get("vehicles", []) or []:
        tc = sym.get("transformed_center")
        if tc is not None:
            mat = sym.get("transform")
            heading = float(np.arctan2(mat[1, 0], mat[0, 0])) if mat is not None else None
            yield np.asarray(tc, dtype=float), heading


def _iter_texts(scene):
    if hasattr(scene, "elements"):
        for t in scene.texts:
            yield np.array([t.position.x, t.position.y]), t.text
        return
    for prim in scene.get("texts", []) or []:
        tc = prim.get("transformed_center")
        if tc is not None:
            yield np.asarray(tc, dtype=float), prim.get("text") or ""


def scene_bounds(scene, pad_frac=0.02) -> tuple[float, float, float, float] | None:
    """(xmin, ymin, xmax, ymax) over all drawable strokes, padded."""
    pts = [xy for xy, _, _ in _iter_strokes(scene)]
    if not pts:
        return None
    allp = np.vstack(pts)
    xmin, ymin = allp.min(axis=0)
    xmax, ymax = allp.max(axis=0)
    pad = pad_frac * max(xmax - xmin, ymax - ymin, 1e-9)
    return xmin - pad, ymin - pad, xmax + pad, ymax + pad


def render_scene(scene, ax=None, mode="annotated", show_texts=False, linewidth=1.0):
    """Render a scene onto a matplotlib Axes (created if not given). Returns the Figure."""
    if ax is None:
        fig = Figure(figsize=(10, 10))
        FigureCanvasAgg(fig)
        ax = fig.add_subplot(111)
    else:
        fig = ax.figure

    strokes_only = mode == "strokes"
    seen_buckets = set()
    for xy, bucket, dashed in _iter_strokes(scene):
        color = "black" if strokes_only else BUCKET_COLORS.get(bucket, "#999999")
        ls = "-" if strokes_only else ("--" if dashed else "-")
        if len(xy) == 1:
            if not strokes_only:
                ax.plot(xy[0, 0], xy[0, 1], ".", color=color, markersize=2)
            continue
        label = None
        if not strokes_only and bucket not in seen_buckets:
            seen_buckets.add(bucket)
            label = bucket
        ax.plot(xy[:, 0], xy[:, 1], ls, color=color, linewidth=linewidth, label=label)

    if not strokes_only:
        for center, heading in _iter_vehicles(scene):
            ax.plot(*center, "o", color=BUCKET_COLORS["vehicle"], markersize=4)
            if heading is not None:
                d = 8.0  # ft
                ax.annotate(
                    "",
                    xy=(center[0] + d * np.cos(heading), center[1] + d * np.sin(heading)),
                    xytext=tuple(center),
                    arrowprops=dict(arrowstyle="->", color=BUCKET_COLORS["vehicle"]),
                )
        if show_texts:
            for pos, text in _iter_texts(scene):
                ax.text(pos[0], pos[1], text, fontsize=6, color="#444444")
        if seen_buckets:
            ax.legend(loc="upper right", fontsize=8)

    ax.set_aspect("equal")
    if strokes_only:
        ax.axis("off")
    return fig


def render_strokes_to_array(scene, size=512, linewidth=1.0) -> np.ndarray | None:
    """
    Rasterize all geometry as black strokes on white into a (size, size) uint8
    array (255 = background). Square framing: the scene bbox is centered and
    letterboxed, matching how vendor thumbs frame the drawing. Returns None
    for scenes with no drawable geometry.
    """
    bounds = scene_bounds(scene)
    if bounds is None:
        return None
    xmin, ymin, xmax, ymax = bounds
    cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
    half = max(xmax - xmin, ymax - ymin) / 2

    dpi = 100
    fig = Figure(figsize=(size / dpi, size / dpi), dpi=dpi)
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_axes([0, 0, 1, 1])
    render_scene(scene, ax=ax, mode="strokes", linewidth=linewidth)
    ax.set_xlim(cx - half, cx + half)
    ax.set_ylim(cy - half, cy + half)

    canvas.draw()
    rgba = np.asarray(canvas.buffer_rgba())
    return rgba[:, :, :3].min(axis=2).astype(np.uint8)

"""
Spline Interpolation Comparison Tool for FARO Zone 2D Polycurve Reverse-Engineering

Compares multiple spline/interpolation methods against FARO Zone polycurve vertices.
Adjust parameters in the CONFIG section below to find the best match.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import (
    CubicSpline, splprep, splev, interp1d,
    BSpline, make_interp_spline, pchip_interpolate
)

# ============================================================================
# RAW DATA - FARO Zone polycurve vertices (x, y, z)
# ============================================================================
RAW = (
    "-168.427772082409,96.2735838096721,4.07972431182861;"
    "-159.821304073235,-4.2176526877434,2.43307089805603;"
    "-147.757427328512,-23.4428512912028,2.13156175613403;"
    "-133.34473680048,-27.8004384045588,2.14206027984619;"
    "-117.33829827654,-17.7245944623032,2.63418626785278;"
    "-110.4681619245,-5.92753414192727,3.03871393203735;"
    "-107.708601059204,15.9750359073449,3.39009189605713;"
    "-112.294365900106,72.976127283281,3.74081373214722"
)

pts = np.array([[float(v) for v in p.split(',')] for p in RAW.split(';')])
x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]

# Parameterize by cumulative chord length (shared across methods)
d = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
t_chord = np.concatenate([[0], np.cumsum(d)])
t_chord /= t_chord[-1]  # normalize to [0, 1]

N_SAMPLES = 500
t_fine = np.linspace(0, 1, N_SAMPLES)


# ============================================================================
# CONFIG - Toggle methods on/off and adjust parameters
# ============================================================================
METHODS = {

    # --- Natural Cubic Spline ---
    # Zero 2nd derivative at endpoints. Most common CAD default.
    "cubic_natural": {
        "enabled": True,
        "color": "#e63946",
        "label": "Cubic Spline (natural)",
        "params": {
            "bc_type": "natural",  # boundary condition
        }
    },

    # --- Not-a-Knot Cubic Spline ---
    # Continuous 3rd derivative at 2nd and penultimate knots.
    # Common alternative CAD default (scipy's default).
    "cubic_not_a_knot": {
        "enabled": True,
        "color": "#457b9d",
        "label": "Cubic Spline (not-a-knot)",
        "params": {
            "bc_type": "not-a-knot",
        }
    },

    # --- Clamped Cubic Spline ---
    # Prescribed first derivatives at endpoints.
    # Set slope values to match expected entry/exit tangents.
    "cubic_clamped": {
        "enabled": True,
        "color": "#2a9d8f",
        "label": "Cubic Spline (clamped)",
        "params": {
            # (dx/dt, dy/dt) at start and end — estimated from first/last segments
            "auto_tangents": True,     # if True, estimates from data; if False, uses manual below
            "start_slope_x": 0.0,      # manual dx/dt at start (only if auto_tangents=False)
            "start_slope_y": -1.0,     # manual dy/dt at start
            "end_slope_x": 0.0,        # manual dx/dt at end
            "end_slope_y": 1.0,        # manual dy/dt at end
        }
    },

    # --- Centripetal Catmull-Rom ---
    # Alpha controls parameterization: 0=uniform, 0.5=centripetal, 1.0=chordal
    # Centripetal avoids cusps and self-intersections.
    "catmull_rom": {
        "enabled": True,
        "color": "#e9c46a",
        "label": "Catmull-Rom (centripetal)",
        "params": {
            "alpha": 0.5,          # 0.0=uniform, 0.5=centripetal, 1.0=chordal
            "endpoint_mode": "extend",  # "extend" = reflect endpoints; "duplicate" = repeat
        }
    },

    # --- Parametric B-Spline (splprep) ---
    # s controls smoothing: 0=exact interpolation, higher=smoother approximation
    # k controls degree: 3=cubic (default), 5=quintic
    "bspline_splprep": {
        "enabled": True,
        "color": "#f4a261",
        "label": "B-Spline (splprep)",
        "params": {
            "s": 0.0,             # smoothing factor: 0=interpolate, try 0.5, 1.0, len(pts)*0.5
            "k": 3,               # spline degree: 3=cubic, 4=quartic, 5=quintic
        }
    },

    # --- B-Spline with make_interp_spline ---
    # Exact interpolation with configurable degree and boundary conditions
    "bspline_interp": {
        "enabled": True,
        "color": "#6a0572",
        "label": "B-Spline (interp, k=3)",
        "params": {
            "k": 3,               # degree: 2=quadratic, 3=cubic, 5=quintic
        }
    },

    # --- PCHIP (Piecewise Cubic Hermite Interpolating Polynomial) ---
    # Monotone-preserving, never overshoots. Good for well-behaved curves.
    # No tunable parameters — included as a monotone reference.
    "pchip": {
        "enabled": True,
        "color": "#264653",
        "label": "PCHIP (monotone cubic)",
        "params": {}
    },

    # --- Akima Spline ---
    # Locally adaptive, less oscillation than natural cubic.
    # Good for data with abrupt changes.
    "akima": {
        "enabled": True,
        "color": "#6b705c",
        "label": "Akima",
        "params": {
            "method": "akima",    # "akima" or "makima" (modified akima, less wiggly)
        }
    },

    # --- Smoothing B-Spline (approximation, not interpolation) ---
    # Good for testing if FARO smooths rather than interpolates exactly.
    "bspline_smooth": {
        "enabled": False,
        "color": "#bc6c25",
        "label": "B-Spline (smoothed)",
        "params": {
            "s_factor": 0.5,      # multiplied by len(pts) for smoothing param
            "k": 3,
        }
    },

    # --- Cubic Bézier Composite ---
    # Fits a composite cubic Bézier through the points.
    # Tension controls how far control points sit from data points.
    "bezier_composite": {
        "enabled": True,
        "color": "#d62828",
        "label": "Composite Bézier",
        "params": {
            "tension": 0.33,      # 0=straight lines, 0.33=default, 0.5=very curved
        }
    },
}


# ============================================================================
# PARAMETERIZATION OPTIONS (applied globally to chord-length methods)
# ============================================================================
PARAMETERIZATION = {
    "method": "chord",    # "chord", "centripetal", "uniform"
    "alpha": 0.5,         # only used for "centripetal": 0.5 is standard
}


# ============================================================================
# INTERPOLATION IMPLEMENTATIONS
# ============================================================================

def compute_parameterization(x, y, method="chord", alpha=0.5):
    """Compute parameter values for the data points."""
    if method == "uniform":
        t = np.linspace(0, 1, len(x))
    elif method == "centripetal":
        d = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
        t = np.concatenate([[0], np.cumsum(d**alpha)])
        t /= t[-1]
    else:  # chord
        d = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
        t = np.concatenate([[0], np.cumsum(d)])
        t /= t[-1]
    return t


def interp_cubic_spline(x, y, t, t_fine, params):
    bc = params["bc_type"]
    cs_x = CubicSpline(t, x, bc_type=bc)
    cs_y = CubicSpline(t, y, bc_type=bc)
    return cs_x(t_fine), cs_y(t_fine)


def interp_cubic_clamped(x, y, t, t_fine, params):
    if params["auto_tangents"]:
        # Estimate tangents from first/last segments, scaled to parameter space
        dt0 = t[1] - t[0]
        dt1 = t[-1] - t[-2]
        sx = [(x[1]-x[0])/dt0, (x[-1]-x[-2])/dt1]
        sy = [(y[1]-y[0])/dt0, (y[-1]-y[-2])/dt1]
    else:
        sx = [params["start_slope_x"], params["end_slope_x"]]
        sy = [params["start_slope_y"], params["end_slope_y"]]

    cs_x = CubicSpline(t, x, bc_type=((1, sx[0]), (1, sx[1])))
    cs_y = CubicSpline(t, y, bc_type=((1, sy[0]), (1, sy[1])))
    return cs_x(t_fine), cs_y(t_fine)


def interp_catmull_rom(x, y, t_fine, params):
    alpha = params["alpha"]
    pts_2d = np.column_stack([x, y])

    # Extend endpoints
    if params["endpoint_mode"] == "extend":
        p_start = 2 * pts_2d[0] - pts_2d[1]
        p_end = 2 * pts_2d[-1] - pts_2d[-2]
    else:  # duplicate
        p_start = pts_2d[0]
        p_end = pts_2d[-1]

    pts_ext = np.vstack([p_start, pts_2d, p_end])

    def tj(ti, pi, pj):
        return ti + np.sqrt((pj[0]-pi[0])**2 + (pj[1]-pi[1])**2)**alpha

    segments = []
    seg_lengths = []
    for i in range(len(pts_ext) - 3):
        p0, p1, p2, p3 = pts_ext[i], pts_ext[i+1], pts_ext[i+2], pts_ext[i+3]
        t0 = 0
        t1 = tj(t0, p0, p1)
        t2 = tj(t1, p1, p2)
        t3 = tj(t2, p2, p3)
        seg_lengths.append(t2 - t1)
        segments.append((p0, p1, p2, p3, t0, t1, t2, t3))

    # Distribute t_fine samples proportional to segment parameter length
    total_len = sum(seg_lengths)
    all_pts = []
    for seg_idx, (p0, p1, p2, p3, t0, t1, t2, t3) in enumerate(segments):
        n_seg = max(int(N_SAMPLES * seg_lengths[seg_idx] / total_len), 10)
        t_local = np.linspace(t1, t2, n_seg, endpoint=(seg_idx == len(segments)-1))

        A1 = np.outer((t1-t_local)/(t1-t0), p0) + np.outer((t_local-t0)/(t1-t0), p1)
        A2 = np.outer((t2-t_local)/(t2-t1), p1) + np.outer((t_local-t1)/(t2-t1), p2)
        A3 = np.outer((t3-t_local)/(t3-t2), p2) + np.outer((t_local-t2)/(t3-t2), p3)
        B1 = np.outer((t2-t_local)/(t2-t0), A1[:, 0]) + np.outer((t_local-t0)/(t2-t0), A2[:, 0])
        B1 = np.column_stack([
            (t2-t_local)/(t2-t0) * A1[:, 0] + (t_local-t0)/(t2-t0) * A2[:, 0],
            (t2-t_local)/(t2-t0) * A1[:, 1] + (t_local-t0)/(t2-t0) * A2[:, 1],
        ])
        B2 = np.column_stack([
            (t3-t_local)/(t3-t1) * A2[:, 0] + (t_local-t1)/(t3-t1) * A3[:, 0],
            (t3-t_local)/(t3-t1) * A2[:, 1] + (t_local-t1)/(t3-t1) * A3[:, 1],
        ])
        C = np.column_stack([
            (t2-t_local)/(t2-t1) * B1[:, 0] + (t_local-t1)/(t2-t1) * B2[:, 0],
            (t2-t_local)/(t2-t1) * B1[:, 1] + (t_local-t1)/(t2-t1) * B2[:, 1],
        ])
        all_pts.append(C)

    result = np.vstack(all_pts)
    return result[:, 0], result[:, 1]


def interp_bspline_splprep(x, y, params):
    tck, u = splprep([x, y], s=params["s"], k=params["k"])
    u_fine = np.linspace(0, 1, N_SAMPLES)
    xf, yf = splev(u_fine, tck)
    return xf, yf


def interp_bspline_interp(x, y, t, t_fine, params):
    k = params["k"]
    spl_x = make_interp_spline(t, x, k=k)
    spl_y = make_interp_spline(t, y, k=k)
    return spl_x(t_fine), spl_y(t_fine)


def interp_pchip(x, y, t, t_fine, params):
    xf = pchip_interpolate(t, x, t_fine)
    yf = pchip_interpolate(t, y, t_fine)
    return xf, yf


def interp_akima(x, y, t, t_fine, params):
    from scipy.interpolate import Akima1DInterpolator
    if params.get("method") == "makima":
        ak_x = Akima1DInterpolator(t, x, method="makima")
        ak_y = Akima1DInterpolator(t, y, method="makima")
    else:
        ak_x = Akima1DInterpolator(t, x)
        ak_y = Akima1DInterpolator(t, y)
    return ak_x(t_fine), ak_y(t_fine)


def interp_bspline_smooth(x, y, params):
    s = params["s_factor"] * len(x)
    tck, u = splprep([x, y], s=s, k=params["k"])
    u_fine = np.linspace(0, 1, N_SAMPLES)
    xf, yf = splev(u_fine, tck)
    return xf, yf


def interp_bezier_composite(x, y, params):
    """Composite cubic Bézier with Hobby-like tangent estimation."""
    tension = params["tension"]
    n = len(x)
    pts_2d = np.column_stack([x, y])

    # Estimate tangents at each point
    tangents = np.zeros_like(pts_2d)
    for i in range(n):
        if i == 0:
            tangents[i] = pts_2d[1] - pts_2d[0]
        elif i == n - 1:
            tangents[i] = pts_2d[-1] - pts_2d[-2]
        else:
            tangents[i] = pts_2d[i+1] - pts_2d[i-1]

    all_pts = []
    for i in range(n - 1):
        p0 = pts_2d[i]
        p3 = pts_2d[i + 1]
        seg_len = np.linalg.norm(p3 - p0)
        p1 = p0 + tension * tangents[i] * seg_len / np.linalg.norm(tangents[i] + 1e-12)
        p2 = p3 - tension * tangents[i+1] * seg_len / np.linalg.norm(tangents[i+1] + 1e-12)

        t_seg = np.linspace(0, 1, max(N_SAMPLES // (n-1), 20))
        curve = (
            np.outer((1-t_seg)**3, p0) +
            np.outer(3*(1-t_seg)**2*t_seg, p1) +
            np.outer(3*(1-t_seg)*t_seg**2, p2) +
            np.outer(t_seg**3, p3)
        )
        all_pts.append(curve if i == 0 else curve[1:])  # avoid duplicating junction points

    result = np.vstack(all_pts)
    return result[:, 0], result[:, 1]


# ============================================================================
# RUN ALL ENABLED METHODS
# ============================================================================

t_param = compute_parameterization(
    x, y,
    method=PARAMETERIZATION["method"],
    alpha=PARAMETERIZATION["alpha"]
)

results = {}
for key, cfg in METHODS.items():
    if not cfg["enabled"]:
        continue
    try:
        p = cfg["params"]
        if key == "cubic_natural" or key == "cubic_not_a_knot":
            xf, yf = interp_cubic_spline(x, y, t_param, t_fine, p)
        elif key == "cubic_clamped":
            xf, yf = interp_cubic_clamped(x, y, t_param, t_fine, p)
        elif key == "catmull_rom":
            xf, yf = interp_catmull_rom(x, y, t_fine, p)
        elif key == "bspline_splprep":
            xf, yf = interp_bspline_splprep(x, y, p)
        elif key == "bspline_interp":
            xf, yf = interp_bspline_interp(x, y, t_param, t_fine, p)
        elif key == "pchip":
            xf, yf = interp_pchip(x, y, t_param, t_fine, p)
        elif key == "akima":
            xf, yf = interp_akima(x, y, t_param, t_fine, p)
        elif key == "bspline_smooth":
            xf, yf = interp_bspline_smooth(x, y, p)
        elif key == "bezier_composite":
            xf, yf = interp_bezier_composite(x, y, p)
        else:
            continue
        results[key] = (xf, yf)
    except Exception as e:
        print(f"  ERROR in {key}: {e}")


# ============================================================================
# PLOT - Overview comparison
# ============================================================================

n_methods = len(results)
fig, axes = plt.subplots(1, 2, figsize=(18, 9))

# Left: all methods overlaid
ax = axes[0]
ax.plot(x, y, 'ko', markersize=8, zorder=10, label='FARO vertices')
for key, (xf, yf) in results.items():
    cfg = METHODS[key]
    ax.plot(xf, yf, color=cfg["color"], linewidth=1.8, alpha=0.85, label=cfg["label"])
ax.set_title("All Methods Overlaid", fontsize=13, fontweight='bold')
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_aspect('equal')
ax.legend(fontsize=8, loc='best')
ax.grid(True, alpha=0.3)

# Right: individual subplots
ax_right = axes[1]
ax_right.axis('off')

n_cols = 3
n_rows = int(np.ceil(n_methods / n_cols))
fig2, axes2 = plt.subplots(n_rows, n_cols, figsize=(18, 5 * n_rows))
if n_rows == 1:
    axes2 = axes2.reshape(1, -1)

for idx, (key, (xf, yf)) in enumerate(results.items()):
    row, col = divmod(idx, n_cols)
    ax2 = axes2[row, col]
    cfg = METHODS[key]
    ax2.plot(x, y, 'ko', markersize=6, zorder=10)
    ax2.plot(xf, yf, color=cfg["color"], linewidth=2)
    ax2.set_title(cfg["label"], fontsize=11, fontweight='bold')
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)

    # Show key params
    param_str = ", ".join(f"{k}={v}" for k, v in cfg["params"].items()
                          if k not in ("auto_tangents",))
    if param_str:
        ax2.text(0.02, 0.02, param_str, transform=ax2.transAxes,
                 fontsize=8, verticalalignment='bottom',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.7))

# Hide unused subplots
for idx in range(len(results), n_rows * n_cols):
    row, col = divmod(idx, n_cols)
    axes2[row, col].axis('off')

fig.tight_layout()
fig2.tight_layout()

fig.savefig('spline_overlay.png', dpi=150, bbox_inches='tight')
fig2.savefig('spline_individual.png', dpi=150, bbox_inches='tight')

print(f"\nGenerated comparison with {n_methods} methods.")
print("Saved: spline_overlay.png, spline_individual.png")
print("\n--- Quick tuning guide ---")
print("Too sharp at turns?  → Increase Catmull-Rom alpha toward 1.0, or try cubic_natural")
print("Too curved/wobbly?   → Try PCHIP or Akima (less oscillation)")
print("Almost right but endpoints off? → Switch cubic_natural ↔ cubic_not_a_knot")
print("Need approximation not interpolation? → Enable bspline_smooth, increase s_factor")
print("FARO likely uses:    → cubic_natural or cubic_not_a_knot (most common for polycurve)")
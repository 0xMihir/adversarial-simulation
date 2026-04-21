"""
Per-diagram normalization (§0.1).

Called identically on synthetic and real FARO scenes — no source-specific logic.
Produces normalized coordinates where:
  - centroid of all vertices is at origin
  - (optionally) principal axis aligned to x-axis via PCA
  - longest convex-hull diagonal = 1.0 unit

The inverse transform is saved alongside each sample so predictions can be
projected back to original metric coordinates at inference time.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.spatial import ConvexHull

from schema.scene import (
    ParsedScene,
    Point2D,
    SceneElement,
    TextElement,
    VehicleDetection,
)


def _pts_to_np(pts: list[Point2D]) -> np.ndarray:
    return np.array([[p.x, p.y] for p in pts], dtype=np.float64)


def _np_to_pts(arr: np.ndarray) -> list[Point2D]:
    return [Point2D(x=float(r[0]), y=float(r[1])) for r in arr]


def _apply_2d(M: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply 3×3 homogeneous affine M to (N, 2) points."""
    if pts.shape[0] == 0:
        return pts
    h = np.ones((pts.shape[0], 3), dtype=np.float64)
    h[:, :2] = pts
    return (M @ h.T).T[:, :2]


class DiagramNormalizer:
    """
    Stateless utility for normalizing ParsedScene geometry.
    All methods are static/classmethod — no instance state.
    """

    @staticmethod
    def collect_all_vertices(scene: ParsedScene) -> np.ndarray:
        """Gather every Point2D from resampled_points across all elements → (N, 2)."""
        parts: list[np.ndarray] = []
        for elem in scene.elements:
            if elem.resampled_points:
                parts.append(_pts_to_np(elem.resampled_points))
        if not parts:
            return np.zeros((0, 2), dtype=np.float64)
        return np.concatenate(parts, axis=0)

    @staticmethod
    def compute_normalization_params(
        vertices: np.ndarray,
        apply_pca: bool = True,
    ) -> tuple[np.ndarray, float, float]:
        """
        Returns:
            centroid: (2,) the mean of hull vertices
            pca_angle: float — rotation angle applied; 0.0 if apply_pca=False
            scale: float — longest hull diagonal in original units (will become 1.0)
        """
        if vertices.shape[0] < 2:
            # Degenerate case: single or no points
            centroid = vertices.mean(axis=0) if vertices.shape[0] > 0 else np.zeros(2)
            return centroid, 0.0, 1.0

        if vertices.shape[0] < 3:
            hull_verts = vertices
        else:
            try:
                hull = ConvexHull(vertices)
                hull_verts = vertices[hull.vertices]
            except Exception:
                hull_verts = vertices

        centroid = hull_verts.mean(axis=0)

        # Longest hull diagonal (max pairwise distance)
        diffs = hull_verts[:, np.newaxis, :] - hull_verts[np.newaxis, :, :]
        dists = np.sqrt((diffs ** 2).sum(axis=-1))
        scale = float(dists.max())
        if scale < 1e-9:
            scale = 1.0

        pca_angle = 0.0
        if apply_pca and hull_verts.shape[0] >= 2:
            centered = hull_verts - centroid
            cov = centered.T @ centered
            # eigh returns eigenvalues ascending; principal axis = last eigenvector
            _, evecs = np.linalg.eigh(cov)
            principal = evecs[:, -1]
            pca_angle = float(math.atan2(principal[1], principal[0]))

        return centroid, pca_angle, scale

    @staticmethod
    def build_forward_transform(
        centroid: np.ndarray,
        pca_angle: float,
        scale: float,
    ) -> np.ndarray:
        """3×3 affine: translate to origin → rotate → scale to unit diagonal."""
        T = np.eye(3)
        T[0, 2] = -centroid[0]
        T[1, 2] = -centroid[1]

        c, s = math.cos(-pca_angle), math.sin(-pca_angle)
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)

        S = np.diag([1.0 / scale, 1.0 / scale, 1.0])

        return S @ R @ T

    @staticmethod
    def build_inverse_transform(
        centroid: np.ndarray,
        pca_angle: float,
        scale: float,
    ) -> np.ndarray:
        """3×3 inverse: unscale → unrotate → translate back."""
        S_inv = np.diag([scale, scale, 1.0])

        c, s = math.cos(pca_angle), math.sin(pca_angle)
        R_inv = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)

        T_inv = np.eye(3)
        T_inv[0, 2] = centroid[0]
        T_inv[1, 2] = centroid[1]

        return T_inv @ R_inv @ S_inv

    @staticmethod
    def apply_transform_to_points(pts: list[Point2D], M: np.ndarray) -> list[Point2D]:
        if not pts:
            return []
        arr = _pts_to_np(pts)
        return _np_to_pts(_apply_2d(M, arr))

    @staticmethod
    def _transform_elem(elem: SceneElement, M: np.ndarray) -> SceneElement:
        new_resampled = _np_to_pts(_apply_2d(M, _pts_to_np(elem.resampled_points))) if elem.resampled_points else []
        new_control = _np_to_pts(_apply_2d(M, _pts_to_np(elem.control_points))) if elem.control_points else []
        new_handles = _np_to_pts(_apply_2d(M, _pts_to_np(elem.bezier_handles))) if elem.bezier_handles else []

        ref_pts = new_resampled or new_control
        if ref_pts:
            xs = [p.x for p in ref_pts]
            ys = [p.y for p in ref_pts]
            new_bbox = (min(xs), min(ys), max(xs), max(ys))
        else:
            new_bbox = elem.bbox

        return elem.model_copy(update={
            "resampled_points": new_resampled,
            "control_points": new_control,
            "bezier_handles": new_handles,
            "bbox": new_bbox,
        })

    @staticmethod
    def _transform_text(text: TextElement, M: np.ndarray) -> TextElement:
        arr = np.array([[text.position.x, text.position.y]], dtype=np.float64)
        tp = _apply_2d(M, arr)[0]
        return text.model_copy(update={"position": Point2D(x=float(tp[0]), y=float(tp[1]))})

    @staticmethod
    def _transform_vehicle(v: VehicleDetection, M: np.ndarray, pca_angle: float = 0.0) -> VehicleDetection:
        new_obb = _np_to_pts(_apply_2d(M, _pts_to_np(v.obb))) if v.obb else []
        center_arr = np.array([[v.center.x, v.center.y]], dtype=np.float64)
        cp = _apply_2d(M, center_arr)[0]
        return v.model_copy(update={
            "obb": new_obb,
            "center": Point2D(x=float(cp[0]), y=float(cp[1])),
            "heading": v.heading - pca_angle,
        })

    @classmethod
    def normalize(
        cls,
        scene: ParsedScene,
        apply_pca: bool = True,
    ) -> tuple[ParsedScene, np.ndarray]:
        """
        Normalize a ParsedScene.

        Returns:
            normalized_scene: deep copy with all geometry in normalized coords
            inverse_transform: (3, 3) np.ndarray mapping normalized → original coords
        """
        vertices = cls.collect_all_vertices(scene)
        centroid, pca_angle, scale = cls.compute_normalization_params(vertices, apply_pca)
        M_fwd = cls.build_forward_transform(centroid, pca_angle, scale)
        M_inv = cls.build_inverse_transform(centroid, pca_angle, scale)

        data = scene.model_dump()
        norm_scene = ParsedScene.model_validate(data)

        norm_scene.elements = [cls._transform_elem(e, M_fwd) for e in scene.elements]
        norm_scene.texts = [cls._transform_text(t, M_fwd) for t in scene.texts]
        norm_scene.vehicles = [cls._transform_vehicle(v, M_fwd, pca_angle) for v in scene.vehicles]

        return norm_scene, M_inv

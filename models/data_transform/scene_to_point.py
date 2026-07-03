"""
ParsedScene -> PTv3 Point conversion (VecFormer line representation).

Each SceneElement (a polyline primitive) is already uniformly sampled into
`resampled_points` by the synthetic generator. Adjacent points are paired into line
segments; each line segment becomes one PTv3 token with:

  coord_i = (cx, cy, 0.0)                          # line midpoint, z=0 (2D -> 3D)
  feat_i  = (l, dx, dy, cx, cy, cpx, cpy)          # VecFormer 7-D line feature

where for a line p1=(x1,y1) -> p2=(x2,y2):
  l          = ||p2 - p1||
  dx, dy     = (x1 - x2)/l, (y1 - y2)/l            # unit displacement
  cx, cy     = midpoint
  cpx, cpy   = primitive centroid = mean of the element's line midpoints

Line tokens are grouped by `primitive_id`; feature_pooling.PrimitivePooling later pools
them into per-primitive tokens, and PrimitiveDecoder classifies each primitive into an
ElementClass.
"""
from __future__ import annotations

from typing import Any, NamedTuple

import numpy as np
import torch

from schema.scene import ParsedScene
from synthetic.normalization import _pts_to_np
from synthetic.schema import ElementClass, SyntheticGroundTruth
from models.primitive_decoder import ELEMENT_CLASSES

FEAT_DIM = 7  # VecFormer line feature; must equal PointTransformerV3 in_channels
_EPS = 1e-9
_OTHER_LABEL = ELEMENT_CLASSES.index(ElementClass.OTHER.value)


def _element_lines(pts: np.ndarray, is_closed: bool) -> np.ndarray | None:
    """Build (p1, p2) line endpoints for one element.

    Returns (L, 4) array of [x1, y1, x2, y2], or None if <1 valid line.
    Drops degenerate (near-zero length) lines.
    """
    if pts.shape[0] < 2:
        return None
    p1 = pts[:-1]
    p2 = pts[1:]
    if is_closed:
        p1 = np.concatenate([p1, pts[-1:]], axis=0)
        p2 = np.concatenate([p2, pts[:1]], axis=0)
    seg = np.concatenate([p1, p2], axis=1)  # (L, 4)
    lengths = np.linalg.norm(seg[:, 2:] - seg[:, :2], axis=1)
    keep = lengths > _EPS
    seg = seg[keep]
    return seg if seg.shape[0] > 0 else None


def _lines_to_feat(seg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert (L, 4) line endpoints to VecFormer feat (L, 7) and coord (L, 3)."""
    p1 = seg[:, :2]
    p2 = seg[:, 2:]
    diff = p1 - p2
    length = np.linalg.norm(diff, axis=1, keepdims=True)  # (L, 1)
    unit = diff / length  # (L, 2) = (dx, dy)
    mid = 0.5 * (p1 + p2)  # (L, 2) = (cx, cy)
    prim_centroid = mid.mean(axis=0, keepdims=True)  # (1, 2) = (cpx, cpy)
    cp = np.broadcast_to(prim_centroid, mid.shape)  # (L, 2)

    feat = np.concatenate([length, unit, mid, cp], axis=1)  # (L, 7)
    coord = np.concatenate([mid, np.zeros((mid.shape[0], 1))], axis=1)  # (L, 3)
    return feat.astype(np.float32), coord.astype(np.float32)


def scene_to_lines(
    scene: ParsedScene,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Convert one ParsedScene into line tokens.

    Returns:
        feat:              (N, 7) VecFormer line features
        coord:             (N, 3) line midpoints (z=0)
        primitive_id:      (N,)   0-based primitive index among kept elements
        valid_element_ids: list[str] element ids in kept order (len == n_prim)
    """
    feats: list[np.ndarray] = []
    coords: list[np.ndarray] = []
    prim_ids: list[np.ndarray] = []
    valid_element_ids: list[str] = []

    prim_idx = 0
    for elem in scene.elements:
        if not elem.resampled_points:
            continue
        pts = _pts_to_np(elem.resampled_points)
        seg = _element_lines(pts, elem.is_closed)
        if seg is None:
            continue
        feat, coord = _lines_to_feat(seg)
        feats.append(feat)
        coords.append(coord)
        prim_ids.append(np.full(feat.shape[0], prim_idx, dtype=np.int64))
        valid_element_ids.append(elem.id)
        prim_idx += 1

    if not feats:
        return (
            np.zeros((0, FEAT_DIM), np.float32),
            np.zeros((0, 3), np.float32),
            np.zeros((0,), np.int64),
            [],
        )

    return (
        np.concatenate(feats, axis=0),
        np.concatenate(coords, axis=0),
        np.concatenate(prim_ids, axis=0),
        valid_element_ids,
    )


def labels_for_scene(
    gt: SyntheticGroundTruth, valid_element_ids: list[str]
) -> np.ndarray:
    """Per-primitive integer labels ordered to match valid_element_ids.

    Maps element_id -> ElementClass value -> ELEMENT_CLASSES index. Unknown ids and
    unknown class strings fall back to OTHER.
    """
    labels = np.empty(len(valid_element_ids), dtype=np.int64)
    for i, eid in enumerate(valid_element_ids):
        cls_val = gt.element_classes.get(eid)
        if cls_val is None or cls_val not in ELEMENT_CLASSES:
            labels[i] = _OTHER_LABEL
        else:
            labels[i] = ELEMENT_CLASSES.index(cls_val)
    return labels


class SampleArrays(NamedTuple):
    """One scene's PTv3 inputs as plain numpy arrays — the small payload a DataLoader
    worker ships to the main process instead of the fat ParsedScene + GroundTruth pydantic
    objects. `feat`/`coord`/`primitive_id` are the per-line arrays from scene_to_lines;
    `labels` is per-primitive, ordered by primitive_id ascending (== valid_ids order).
    """

    feat: np.ndarray  # (N, 7) float32
    coord: np.ndarray  # (N, 3) float32
    primitive_id: np.ndarray  # (N,) int64
    labels: np.ndarray  # (n_prim,) int64


def scene_to_arrays(scene: ParsedScene, gt: SyntheticGroundTruth) -> SampleArrays:
    """Turn one (scene, gt) into the small numpy arrays PTv3 needs.

    Intended to run inside the DataLoader worker so only these arrays (not the ~1.5MB
    pydantic scene+gt) cross the IPC boundary. batch_point_dict assembles a batch of these.
    """
    feat, coord, prim_id, valid_ids = scene_to_lines(scene)
    labels = labels_for_scene(gt, valid_ids)
    return SampleArrays(feat=feat, coord=coord, primitive_id=prim_id, labels=labels)


def batch_point_dict(
    samples: list[SampleArrays], grid_size: float = 0.01
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Assemble per-sample SampleArrays into a PTv3 data_dict + per-primitive labels.

    This is the second half of the old build_point_dict: it only concatenates cheap numpy
    arrays (no scene_to_lines work), so it's fine to run in the main process on the already
    worker-produced samples.

    Returns:
        data_dict: dict with coord (N,3), feat (N,7), batch (N,), grid_size (scalar),
                   primitive_id (N,), nprim_per_sample (B,).
        labels:    (sum nprim,) long, ordered per-sample then primitive_id ascending,
                   matching PrimitivePooling's output row order.
    """
    feats: list[np.ndarray] = []
    coords: list[np.ndarray] = []
    prim_ids: list[np.ndarray] = []
    batch_idx: list[np.ndarray] = []
    label_parts: list[np.ndarray] = []
    nprim_per_sample: list[int] = []

    for b, s in enumerate(samples):
        feats.append(s.feat)
        coords.append(s.coord)
        prim_ids.append(s.primitive_id)
        batch_idx.append(np.full(s.feat.shape[0], b, dtype=np.int64))
        label_parts.append(s.labels)
        nprim_per_sample.append(int(s.labels.shape[0]))

    data_dict = {
        "coord": torch.from_numpy(np.concatenate(coords, axis=0)),
        "feat": torch.from_numpy(np.concatenate(feats, axis=0)),
        "batch": torch.from_numpy(np.concatenate(batch_idx, axis=0)),
        "primitive_id": torch.from_numpy(np.concatenate(prim_ids, axis=0)),
        "nprim_per_sample": torch.tensor(nprim_per_sample, dtype=torch.long),
        "grid_size": grid_size,
    }
    labels = torch.from_numpy(np.concatenate(label_parts, axis=0))
    return data_dict, labels


def build_point_dict(
    batch: dict[str, Any], grid_size: float = 0.01
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Collate a raw-scene dataset batch into a PTv3 data_dict + per-primitive labels.

    Kept for the non-worker path (e.g. tests, or a Dataset built with transform_in_worker
    off). Runs scene_to_lines in the caller. When the dataset transforms in the worker,
    use batch_point_dict on the pre-built arrays instead.

    Args:
        batch: output with keys "scenes" and "ground_truths".
        grid_size: quantization size required by Point.serialization / sparsify.
    """
    scenes = batch["scenes"]
    gts = batch["ground_truths"]
    samples = [scene_to_arrays(scene, gt) for scene, gt in zip(scenes, gts)]
    return batch_point_dict(samples, grid_size=grid_size)

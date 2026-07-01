from typing import Literal

import torch
import torch.nn as nn
from torch_scatter import segment_csr

from models.pointcept.models import PointModule
from models.pointcept.models.utils.structure import Point


def build_indptr(
    prim_local: torch.Tensor,   # (N,) per-sample primitive index for each point
    batch: torch.Tensor,        # (N,) which sample each point belongs to
    nprim_per_sample: torch.Tensor,  # (B,) primitive count per sample
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns (prim_global, indptr) where prim_global is a globally-unique primitive
    index across the batch and indptr is the CSR boundary array for segment_csr.

    If points are not already laid out primitive-contiguously, the caller must sort
    by prim_global before calling segment_csr.
    """
    prim_offset = torch.cat(
        [nprim_per_sample.new_zeros(1), nprim_per_sample.cumsum(0)]
    )  # (B+1,)
    prim_global = prim_local + prim_offset[batch]  # (N,) unique across batch
    n_prim = int(prim_offset[-1])
    counts = torch.bincount(prim_global, minlength=n_prim)  # (n_prim,)
    indptr = torch.cat([counts.new_zeros(1), counts.cumsum(0)])  # (n_prim+1,)
    return prim_global, indptr


class AttnPool(nn.Module):
    """Segment-softmax attention pooling over primitives."""

    def __init__(self, dim: int):
        super().__init__()
        self.score = nn.Linear(dim, 1)

    def forward(self, feat: torch.Tensor, indptr: torch.Tensor) -> torch.Tensor:
        sizes = indptr.diff()  # (n_prim,)
        w = self.score(feat)  # (N, 1)
        seg_max = segment_csr(w, indptr, reduce="max")  # (n_prim, 1)
        # Subtract per-segment max for numerical stability before softmax
        w = w - seg_max.repeat_interleave(sizes, dim=0)
        e = w.exp()
        denom = segment_csr(e, indptr, reduce="sum").repeat_interleave(sizes, dim=0)
        a = e / denom  # segment softmax weights
        return segment_csr(feat * a, indptr, reduce="sum")  # (n_prim, dim)


class PrimitivePooling(PointModule):
    """
    Pool per-point features (from PTv3 decoder) into per-primitive tokens.

    Expects the input Point to carry:
      - feat:             (N, dim) per-point features
      - batch:            (N,)     sample index for each point
      - primitive_id:     (N,)     per-sample primitive index (0-based, local to each sample)
      - nprim_per_sample: (B,)     number of primitives in each sample

    primitive_id must be globally unique across the batch after offsetting by
    nprim_per_sample cumsum — build_indptr handles this.  Points do *not* need to
    be sorted by primitive beforehand; this module sorts internally when needed.

    Output: (n_prim, dim) tensor of primitive tokens, where n_prim = sum(nprim_per_sample).
    """

    def __init__(self, pooling_fn: Literal["mean", "max", "attn"], dim: int = 64):
        super().__init__()
        if pooling_fn not in ("mean", "max", "attn"):
            raise ValueError("pooling_fn must be one of ['mean', 'max', 'attn']")
        self.pooling_fn = pooling_fn
        self.attn_pool = AttnPool(dim) if pooling_fn == "attn" else None

    def forward(self, point: Point) -> torch.Tensor:
        feat = point.feat                        # (N, dim)
        batch = point.batch                      # (N,)
        prim_local = point.primitive_id          # (N,)
        nprim_per_sample = point.nprim_per_sample  # (B,)

        assert feat.shape[0] == prim_local.shape[0], (
            f"feat and primitive_id length mismatch: {feat.shape[0]} vs {prim_local.shape[0]}"
        )

        prim_global, indptr = build_indptr(prim_local, batch, nprim_per_sample)

        # Sort points into primitive-contiguous order if not already sorted
        # segment_csr requires rows to be laid out [prim0_pts | prim1_pts | ...]
        sorted_order = prim_global.argsort(stable=True)
        feat_sorted = feat[sorted_order]

        if self.pooling_fn == "attn":
            assert self.attn_pool is not None
            return self.attn_pool(feat_sorted, indptr)
        return segment_csr(feat_sorted, indptr, reduce=self.pooling_fn)  # (n_prim, dim)

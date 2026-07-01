"""
scatter_nd.py — a faithful, dependency-light (numpy-only) reimplementation of
torch_scatter.scatter, written to make the role of `dim` obvious.

Semantics mirror https://pytorch-scatter.readthedocs.io/en/latest/functions/scatter.html

Key invariant: scatter touches exactly ONE axis (`dim`). The output has the same
shape as `src` on every axis except `dim`, whose size becomes `dim_size`
(or index.max()+1). For every element of `src`, its output position equals its
own multi-index EXCEPT along `dim`, where the position is taken from `index`.
Elements that land on the same output slot are combined by `reduce`.
"""

import numpy as np
from itertools import product


# ---------------------------------------------------------------------------
# core
# ---------------------------------------------------------------------------

def _reduce(vals, op):
    """Reduce a list of contributing values. Empty slot -> reduction identity,
    matching torch_scatter (sum/mean/min/max -> 0, mul -> 1)."""
    if len(vals) == 0:
        return 1.0 if op == "mul" else 0.0
    if op == "sum":
        return float(np.sum(vals))
    if op == "mul":
        return float(np.prod(vals))
    if op == "mean":
        return float(np.mean(vals))
    if op == "min":
        return float(np.min(vals))
    if op == "max":
        return float(np.max(vals))
    raise ValueError(f"unknown reduce: {op!r}")


def broadcast_index(index, src_shape, dim):
    """Replicate torch_scatter's broadcasting rule.

    A 1-D index is placed ALONG `dim` (length must equal src.shape[dim]) and
    broadcast across every other axis. An index that already matches src.ndim
    is broadcast normally. The returned array has the full src shape.
    """
    index = np.asarray(index)
    ndim = len(src_shape)
    if index.ndim == 1:
        # put the 1-D index on axis `dim`, singletons elsewhere
        shape = [1] * ndim
        shape[dim] = index.shape[0]
        index = index.reshape(shape)
    return np.broadcast_to(index, src_shape)


def scatter(src, index, dim=-1, dim_size=None, reduce="sum"):
    """Pure-numpy torch_scatter.scatter."""
    src = np.asarray(src, dtype=float)
    dim = dim % src.ndim
    index = broadcast_index(index, src.shape, dim).astype(int)

    y = dim_size if dim_size is not None else int(index.max()) + 1
    out_shape = list(src.shape)
    out_shape[dim] = y

    # bucket every src element into its output slot, then reduce
    contrib = {}
    for p in product(*[range(s) for s in src.shape]):
        out_idx = list(p)
        out_idx[dim] = int(index[p])
        contrib.setdefault(tuple(out_idx), []).append(src[p])

    out = np.empty(out_shape, dtype=float)
    for oi in product(*[range(s) for s in out_shape]):
        out[oi] = _reduce(contrib.get(oi, []), reduce)
    return out


# ---------------------------------------------------------------------------
# demo
# ---------------------------------------------------------------------------

def banner(t):
    print("\n" + "=" * 68)
    print(t)
    print("=" * 68)


def show_dim_effect():
    banner("Same 3-D src, scattered along each axis (reduce='sum')")
    src = np.arange(24, dtype=float).reshape(2, 3, 4)
    print("src.shape =", src.shape)
    print(src.astype(int))

    # one index per axis, each collapsing that axis to 2 buckets
    idx = {0: [0, 0],          # axis 0 (len 2): both -> bucket 0
           1: [0, 1, 0],       # axis 1 (len 3): rows 0,2 -> 0 ; row 1 -> 1
           2: [0, 0, 1, 1]}    # axis 2 (len 4): cols 0,1 -> 0 ; cols 2,3 -> 1

    for d in (0, 1, 2):
        out = scatter(src, idx[d], dim=d, reduce="sum")
        print(f"\n--- dim={d}, index={idx[d]} ---")
        print(f"shape {src.shape}  --dim={d}-->  {out.shape}   "
              f"(only axis {d} changes size)")
        print(out.astype(int))


def show_reduce_effect():
    banner("Fixed dim=1, varying reduce (collapses axis-1 from 3 -> 2)")
    src = np.array([[[1., 5.], [4., 2.], [3., 9.]],
                    [[6., 0.], [7., 8.], [2., 1.]]])  # shape (2,3,2)
    index = [0, 1, 0]  # rows 0,2 share bucket 0 ; row 1 -> bucket 1
    print("src.shape =", src.shape, "  index =", index, "  dim=1")
    print("buckets along axis 1: {0: rows[0,2], 1: rows[1]}\n")
    for op in ("sum", "mul", "mean", "min", "max"):
        out = scatter(src, index, dim=1, reduce=op)
        print(f"reduce={op:<5}  out.shape={out.shape}")
        print(out, "\n")


def show_dim_size_and_broadcast():
    banner("dim_size (force width) and 1-D index broadcasting")
    src = np.arange(12, dtype=float).reshape(3, 4)  # (3,4)
    index = [0, 2, 2, 0]                            # along dim=1 (len 4)
    auto = scatter(src, index, dim=1, reduce="sum")
    forced = scatter(src, index, dim=1, dim_size=5, reduce="sum")
    print("src (3,4):"); print(src.astype(int))
    print(f"\nindex={index} broadcast across all 3 rows; bucket 1 is empty.")
    print("auto   (dim_size=index.max()+1=3):", auto.shape)
    print(auto.astype(int))
    print("\nforced (dim_size=5) -> trailing empty buckets appear as 0:", forced.shape)
    print(forced.astype(int))


if __name__ == "__main__":
    show_dim_effect()
    show_reduce_effect()
    show_dim_size_and_broadcast()
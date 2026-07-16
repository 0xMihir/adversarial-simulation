"""
SeqGrowGraph token vocabulary — single source of truth for the sequence layout.

Mirrors the paper's vocabulary (arXiv 2507.04822 §3.3) with bin ranges adapted to this
repo's normalized scene coordinates (centroid-centered, longest hull diagonal = 1.0,
so coords live in roughly [-0.6, 0.6]):

  [0, N_COORD_BINS)                       node-position coordinate bins
  [NODE_INDEX_BASE, NODE_INDEX_BASE+MAX_NODES)   node-index tokens (DFS index i -> BASE+i)
  [CTRL_BASE, CTRL_BASE+N_CTRL_BINS)      Bezier middle-ctrl-point coordinate bins
  TOK_TO / TOK_SEP / TOK_EOS / TOK_BOS / TOK_PAD  specials

The ctrl-point range is wider than the node range because a quadratic Bezier's middle
control point can lie outside the hull of its endpoints.
"""
from __future__ import annotations

import numpy as np

COORD_MIN, COORD_MAX = -0.65, 0.65  # node coords (~[-0.6,0.6] observed + margin)
N_COORD_BINS = 200  # tokens 0..199

NODE_INDEX_BASE = 200
MAX_NODES = 150  # index tokens 200..349

CTRL_BASE = 350
N_CTRL_BINS = 220  # tokens 350..569
CTRL_MIN, CTRL_MAX = -0.975, 0.975

TOK_TO = 570
TOK_SEP = 571
TOK_EOS = 572
TOK_BOS = 573
TOK_PAD = 574
VOCAB_SIZE = 575

MAX_SEQ_LEN = 768  # incl. BOS/EOS; encode_graph truncates at a <SEP> boundary

assert NODE_INDEX_BASE == N_COORD_BINS
assert CTRL_BASE == NODE_INDEX_BASE + MAX_NODES
assert TOK_TO == CTRL_BASE + N_CTRL_BINS
assert VOCAB_SIZE == TOK_PAD + 1


def _to_bin(v: np.ndarray | float, lo: float, hi: float, n: int) -> np.ndarray:
    """Quantize value(s) in [lo, hi] to integer bins 0..n-1 (clamped)."""
    x = np.asarray(v, dtype=np.float64)
    b = np.floor((x - lo) / (hi - lo) * n).astype(np.int64)
    return np.clip(b, 0, n - 1)


def _from_bin(b: np.ndarray | int, lo: float, hi: float, n: int) -> np.ndarray:
    """Dequantize bins to bin-center values."""
    x = np.asarray(b, dtype=np.float64)
    return lo + (x + 0.5) / n * (hi - lo)


def coord_to_bin(v: np.ndarray | float) -> np.ndarray:
    return _to_bin(v, COORD_MIN, COORD_MAX, N_COORD_BINS)


def bin_to_coord(b: np.ndarray | int) -> np.ndarray:
    return _from_bin(b, COORD_MIN, COORD_MAX, N_COORD_BINS)


def ctrl_to_bin(v: np.ndarray | float) -> np.ndarray:
    """Ctrl-point value -> token id (already offset by CTRL_BASE)."""
    return _to_bin(v, CTRL_MIN, CTRL_MAX, N_CTRL_BINS) + CTRL_BASE


def bin_to_ctrl(b: np.ndarray | int) -> np.ndarray:
    """Token id in the ctrl range -> ctrl-point value."""
    return _from_bin(np.asarray(b) - CTRL_BASE, CTRL_MIN, CTRL_MAX, N_CTRL_BINS)

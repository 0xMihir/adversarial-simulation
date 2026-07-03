"""
Primitive classification decoder.

Consumes per-primitive tokens (n_prim, dim) produced by feature_pooling and predicts
one ElementClass per primitive. VecFormer-style: the pooled token summarizes all the
line segments belonging to a primitive; this head maps it to class logits.
"""
import torch
import torch.nn as nn
from timm.layers import trunc_normal_

from synthetic.schema import ElementClass

# Canonical ordered class list. Built from the enum so the integer <-> class mapping
# stays in sync with the data pipeline; label assembly must use ELEMENT_CLASSES.index(...).
ELEMENT_CLASSES: tuple[str, ...] = tuple(c.value for c in ElementClass)
NUM_ELEMENT_CLASSES: int = len(ELEMENT_CLASSES)


class PrimitiveDecoder(nn.Module):
    """MLP classification head over per-primitive tokens.

    Input:  (n_prim, dim) pooled primitive tokens.
    Output: (n_prim, num_classes) class logits (no softmax).
    """

    def __init__(
        self,
        dim: int = 64,
        hidden_dim: int = 128,
        num_classes: int = NUM_ELEMENT_CLASSES,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.mlp(tokens)  # (n_prim, num_classes)

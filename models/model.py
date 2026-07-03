from torch import nn

# Import the base (m1) variant explicitly: the package __init__ re-exports m1/m2/m3 in
# order, so `PointTransformerV3` from the package would resolve to m3 (utonia).
from pointcept.models.point_transformer_v3.point_transformer_v3m1_base import (
    PointTransformerV3,
)
from models.feature_pooling import PrimitivePooling
from models.primitive_decoder import PrimitiveDecoder
from models.data_transform.scene_to_point import FEAT_DIM


class Model(nn.Module):
    """PTv3 backbone -> per-primitive pooling -> ElementClass classification.

    forward(data_dict) expects a Pointcept-style dict carrying, beyond the usual
    coord/feat/batch/grid_size, the per-line `primitive_id` (N,) and `nprim_per_sample`
    (B,) used by PrimitivePooling. See models.data_transform.build_point_dict.

    Returns per-primitive class logits (n_prim, num_classes).
    """

    def __init__(self, embed_dim: int = 64):
        super().__init__()
        self.backbone = PointTransformerV3(in_channels=FEAT_DIM)
        self.pooling = PrimitivePooling(pooling_fn="maxavg", dim=embed_dim)
        self.decoder = PrimitiveDecoder(dim=embed_dim)

    def forward(self, data_dict):
        point = self.backbone(data_dict)  # Point, feat (N, embed_dim)
        # primitive_id / nprim_per_sample ride along on the Point (addict.Dict) but are
        # not always propagated by backbone stages; re-attach from the input dict.
        point.primitive_id = data_dict["primitive_id"]
        point.nprim_per_sample = data_dict["nprim_per_sample"]
        tokens = self.pooling(point)  # (n_prim, embed_dim)
        return self.decoder(tokens)  # (n_prim, num_classes)

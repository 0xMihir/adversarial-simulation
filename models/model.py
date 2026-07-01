from torch import nn
from pointcept.models.point_transformer_v3 import PointTransformerV3
from feature_pooling import PrimitivePooling

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.backbone = PointTransformerV3()
        self.avg_pooling = PrimitivePooling(pooling_fn='mean')
        
        
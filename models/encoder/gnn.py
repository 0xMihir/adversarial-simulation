from torch  import nn
import torch
from torch_geometric.nn import GAT
from schema import ParsedScene, SceneElement



class GNNEncoder(nn.Module):
    def __init__(self, scene: ParsedScene, embedding_dim = 64, num_hidden_layers = 6):
        self.scene = scene
        self.gat1 = GAT()

    def forward(self, x, edge_index):
        x1 = self.sequential(x, edge_index)
        

        
        
        
        
        


import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing

class WeatherMessageRouter(MessagePassing):
    """
    Custom DeepMind-style Message Passing Layer.
    Routes data features across distinct topological mesh dimensions.
    """
    def __init__(self, in_src, in_tgt, out_dim):
        # Using 'mean' aggregation to stabilize atmospheric feature calculations
        super(WeatherMessageRouter, self).__init__(aggr='mean')
        
        # Linear layers to process node characteristics paired with edge distances
        self.msg_mlp = nn.Sequential(
            nn.Linear(in_src + in_tgt + 1, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim)
        )
        self.update_layer = nn.Linear(out_dim + in_tgt, out_dim)

    def forward(self, x_src, x_tgt, edge_index, edge_dist):
        return self.propagate(edge_index, x=(x_src, x_tgt), edge_dist=edge_dist, 
                              size=(x_src.size(0), x_tgt.size(0)))

    def message(self, x_i, x_j, edge_dist):
        # x_i: Destination landscape coordinates, x_j: Source landscape coordinates
        return self.msg_mlp(torch.cat([x_i, x_j, edge_dist], dim=-1))

    def update(self, aggr_out, x):
        # CORRECTION: Extract the target node tensor matrix explicitly from the tuple structure
        x_src, x_tgt = x
        return torch.relu(self.update_layer(torch.cat([aggr_out, x_tgt], dim=-1)))
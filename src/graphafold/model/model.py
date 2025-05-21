import torch
from dgl.nn import NNConv
import torch.nn as nn
import torch.nn.functional as F


class GNNModel(nn.Module):
    def __init__(self, in_feats, edge_feats, hidden_feats, out_feats):
        super(GNNModel, self).__init__()
        self.node_proj = nn.Linear(in_feats, hidden_feats)
        edge_nn = nn.Sequential(
            nn.Linear(edge_feats, hidden_feats * hidden_feats),
            nn.ReLU()
        )
        self.conv1 = NNConv(hidden_feats, hidden_feats, edge_nn, aggregator_type='mean')
        self.conv2 = NNConv(hidden_feats, hidden_feats, edge_nn, aggregator_type='mean')
        self.edge_predictor = nn.Sequential(
            nn.Linear(hidden_feats * 2, hidden_feats),
            nn.ReLU(),
            nn.Linear(hidden_feats, out_feats)
        )

    def forward(self, g, edge_list):
        h = g.ndata['feat']
        e = g.edata['feat']
        h = self.node_proj(h)
        h = self.conv1(g, h, e)
        h = F.relu(h)
        h = self.conv2(g, h, e)
        h = F.relu(h)
        src_nodes = edge_list[:, 0]
        dst_nodes = edge_list[:, 1]
        src_h = h[src_nodes]
        dst_h = h[dst_nodes]
        edge_inputs = torch.cat([src_h, dst_h], dim=1)
        logits = self.edge_predictor(edge_inputs)
        return logits

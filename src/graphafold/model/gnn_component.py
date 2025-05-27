from dgl.nn import NNConv
import torch.nn as nn
import torch.nn.functional as F


class GNNModel(nn.Module):
    def __init__(self, in_feats, edge_feats, hidden_feats):
        super(GNNModel, self).__init__()
        self.node_proj = nn.Linear(in_feats, hidden_feats) # tu jest tylko funkcja liniowa, ktora bez aktywacji niewiele zmienia
        edge_nn = nn.Sequential(
            nn.Linear(edge_feats, hidden_feats * hidden_feats),
            nn.ReLU()
        )
        self.conv1 = NNConv(hidden_feats, hidden_feats, edge_nn, aggregator_type='mean')
        self.conv2 = NNConv(hidden_feats, hidden_feats, edge_nn, aggregator_type='mean')

    def forward(self, g, node_features, edge_features=None):
        """
        Forward pass through the GNN model.
        Args:
            g (DGLGraph): Input graph.
            node_features (torch.Tensor, optional): Node features. If None, uses the graph's node features.
            edge_features (torch.Tensor, optional): Edge features. If None, uses the graph's edge features.
        Returns:
            torch.Tensor: Output node features after GNN layers.
        """
        h = node_features # g.ndata['feat']
        if edge_features is None:
            e = g.edata['feat']
        else:
            e = edge_features
        h = self.node_proj(h)
        h = self.conv1(g, h, e)
        h = F.relu(h)
        h = self.conv2(g, h, e)
        h = F.relu(h)
        return h
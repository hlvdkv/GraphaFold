from dgl.nn import NNConv
import torch.nn as nn
import torch.nn.functional as F


class GNNModel(nn.Module):
    def __init__(self, in_feats, edge_feats, hidden_feats, gcn_layers=2):
        """
        Initialize the GNN model.
        Args:
            in_feats (int): Number of input features for each node.
            edge_feats (int): Number of features for each edge.
            hidden_feats (int): Number of hidden features for each node.
            gcn_layers (int): Number of GCN layers to use.
        """
        super(GNNModel, self).__init__()
        self.node_proj = nn.Linear(in_feats, hidden_feats)
        self.edge_nn = nn.Sequential(
            nn.Linear(edge_feats, hidden_feats * hidden_feats),  # Assuming edge features are of size 2
            nn.ReLU(),
        )                
        self.convs = nn.ModuleList([
            NNConv(hidden_feats, hidden_feats, self.edge_nn, aggregator_type='mean') for _ in range(gcn_layers)
        ])

    def forward(self, g, node_features):
        """
        Forward pass through the GNN model.
        Args:
            g (DGLGraph): Input graph.
            node_features (torch.Tensor, optional): Node features. If None, uses the graph's node features.
            edge_features (torch.Tensor, optional): Edge features. If None, uses the graph's edge features.
        Returns:
            torch.Tensor: Output node features after GNN layers.
        """
        g.ndata['feat'] = node_features
        h = g.ndata['feat']
        e = g.edata['feat']
        h = self.node_proj(h)
        for conv in self.convs:
            h = conv(g, h, e)
            h = F.relu(h)
        return h
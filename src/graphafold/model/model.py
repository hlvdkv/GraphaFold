import dgl
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from graphafold.model import GNNModel
from rinalmo.pretrained import get_pretrained_model


class Model(nn.Module):
    def __init__(self, in_feats, edge_feats, hidden_feats, hidden_dim:int=384):
        super(Model, self).__init__()
        self.gnn_model = GNNModel(in_feats, edge_feats, hidden_feats)
        self.rinalmo, self.alphabet = get_pretrained_model(model_name="giga-v1")
        self.sequence_embedder = nn.Sequential([
                                    nn.Linear(1280, hidden_dim),  # RiNALMo embedding dim is 1280
                                    nn.ReLU(),
                                    nn.Linear(hidden_dim, hidden_feats)
                                    ])

        self.classifier = nn.Sequential(
            nn.Linear(hidden_feats * 2, hidden_feats),
            nn.ReLU(),
            nn.Linear(hidden_feats, 1),  # Probability score for edge existence
        )
        self.hidden_feats = hidden_feats

    def forward(self, sequence:Tensor[str], g:dgl.DGLGraph, edge_candidates:Tensor) -> torch.Tensor:
        """
        Forward pass through the model.
        Args:
            sequence (str): Input sequence to be processed by RiNALMo.
        Returns:
            torch.Tensor: Output logits from the GNN model.
        """
        self.rinalmo.eval()  # Set RiNALMo to evaluation mode
        tokens = torch.tensor(self.alphabet.batch_tokenize(sequence), dtype=torch.int64)
        with torch.no_grad(), torch.cuda.amp.autocast():
            out_representation = self.rinalmo(tokens)["representation"]
        sequence_embedding = self.sequence_embedder(out_representation)

        # set features in nodes in the graph
        node_embeddings = self.gnn_model(g, node_features=sequence_embedding)
        src = node_embeddings[edge_candidates[0]]  # [K, dim]
        dst = node_embeddings[edge_candidates[1]]  # [K, dim]
        # dot-product for scores
        # scores = src * dst
        # Concatenate and classify
        combined = torch.cat([src, dst], dim=-1)  # [K, 2 * dim]
        logits = self.classifier(combined).squeeze(-1)  # [K]
        return logits






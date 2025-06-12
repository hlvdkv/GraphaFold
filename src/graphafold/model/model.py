import dgl
import lightning as L
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from graphafold.model.gnn_component import GNNModel
from rinalmo.pretrained import get_pretrained_model


class GraphaFold(L.LightningModule):
    def __init__(self,
                 in_feats:int=512,
                 edge_feats:int=128,
                 hidden_feats:int=256,
                 hidden_dim:int=384,
                 gcn_layers:int=2):
        super(GraphaFold, self).__init__()
        self.gnn_model = GNNModel(hidden_feats, edge_feats, hidden_feats, gcn_layers)
        self.rinalmo, self.alphabet = get_pretrained_model(model_name="giga-v1")

        self.sequence_embedder = nn.Sequential(
            nn.Linear(1280, hidden_feats),  # RiNALMo embedding dim is 1280
            nn.ReLU(),
            nn.Linear(hidden_feats, hidden_feats),
            nn.ReLU(),
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_feats * 2, hidden_feats),
            nn.ReLU(),
            nn.Linear(hidden_feats, 1),  # Probability score for edge existence
        )
        self.hidden_feats = hidden_feats

    def forward(self, g:dgl.DGLGraph, sequence:Tensor, edge_candidates:Tensor) -> torch.Tensor:
        """
        Forward pass through the model.
        Args:
            sequence (str): Input sequence to be processed by RiNALMo.
        Returns:
            torch.Tensor: Output logits from the GNN model.
        """
        self.rinalmo.eval()  # Set RiNALMo to evaluation mode
        
        tokens = torch.tensor(self.alphabet.batch_tokenize(sequence), dtype=torch.int64, device=self.device)
        # rna positions in flatten tokens are in places where token value > 4
        flat_tokens = tokens.flatten()
        # ignore special tokens and keep only RNA positions - values are in range 4-16
        rna_positions = torch.where((flat_tokens > 4) & (flat_tokens < 16))[0]

        with torch.no_grad(), torch.cuda.amp.autocast():
            out_representation = self.rinalmo(tokens)["representation"]
        sequence_embedding = self.sequence_embedder(out_representation)
        sequence_embedding = sequence_embedding.reshape(-1, sequence_embedding.shape[-1])  # Flatten to [batch_size * seq_len, hidden_feats]
        sequence_embedding = sequence_embedding[rna_positions] # Keep only RNA positions

        # set features in nodes in the graph
        assert g.num_nodes() == sequence_embedding.shape[0], \
            f"Graph nodes {g.num_nodes()} do not match sequence embedding size {sequence_embedding.shape[0]} for {sequence}"
        node_embeddings = self.gnn_model(g, node_features=sequence_embedding)
        src = node_embeddings[edge_candidates[:, 0]]
        dst = node_embeddings[edge_candidates[:, 1]]
        # dot-product for scores
        # scores = src * dst
        # Concatenate and classify
        combined = torch.cat([src, dst], dim=-1)  # [K, 2 * dim]
        logits = self.classifier(combined).squeeze(-1)  # [K]
        return logits
    
    def training_step(self, batch, batch_idx):
        g, sequence, edge_candidates, labels = batch
        logits = self(g, sequence, edge_candidates)
        # loss = F.binary_cross_entropy_with_logits(logits, labels.float())
        pos_weight = torch.tensor([10.0], device=self.device)  # Adjust pos_weight based on your dataset
        loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)(logits, labels.float())
        # loss = self.focal_loss(logits, labels.float())
        metrics = self.metrics(logits, labels)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log_dict(metrics, on_step=True, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        g, sequence, edge_candidates, labels = batch
        logits = self(g, sequence, edge_candidates)
        # loss = F.binary_cross_entropy_with_logits(logits, labels.float())
        pos_weight = torch.tensor([10.0], device=self.device)  # Adjust pos_weight based on your dataset
        loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)(logits, labels.float())
        # loss = self.focal_loss(logits, labels.float())
        metrics = self.metrics(logits, labels, prefix="val_")
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log_dict(metrics, on_step=True, on_epoch=True, prog_bar=True)
        return loss
    
    def metrics(self, logits, labels, prefix=""):
        """ Calculate precision, recall, F1-score and accuracy for the predictions. """
        preds = torch.sigmoid(logits) > 0.8  # Convert logits to binary predictions
        tp = (preds & labels.bool()).sum().item()
        fp = (preds & ~labels.bool()).sum().item()
        fn = (~preds & labels.bool()).sum().item()
        tn = (~preds & ~labels.bool()).sum().item()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0.0
        return {
            f"{prefix}precision": precision,
            f"{prefix}recall": recall,
            f"{prefix}f1_score": f1_score,
            f"{prefix}accuracy": accuracy,
        }
    
    def focal_loss(self, logits, targets, alpha=0.25, gamma=2.0):
        BCE_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        probs = torch.sigmoid(logits)
        pt = torch.where(targets == 1, probs, 1 - probs)
        focal_factor = (1 - pt) ** gamma
        alpha_factor = torch.where(targets == 1, alpha, 1 - alpha)
        return (alpha_factor * focal_factor * BCE_loss).mean()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
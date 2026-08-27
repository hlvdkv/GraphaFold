from __future__ import annotations

from contextlib import nullcontext

import torch
from torch import nn

from .data import RNAGraph

class RelationalGraphBlock(nn.Module):
    """Edge-type-aware gated message passing with a residual update."""

    def __init__(self, dim: int, num_edge_types: int = 2, dropout: float = 0.1) -> None:
        super().__init__()
        self.dim = dim
        self.message_weights = nn.Parameter(torch.empty(num_edge_types, dim, dim))
        nn.init.xavier_uniform_(self.message_weights)
        self.edge_embedding = nn.Embedding(num_edge_types, dim)
        self.pre_norm = nn.LayerNorm(dim)
        self.gate = nn.Sequential(
            nn.Linear(dim * 3, dim),
            nn.GELU(),
            nn.Linear(dim, 1),
        )
        self.update = nn.Sequential(
            nn.Linear(dim * 2, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
        )
        self.dropout = nn.Dropout(dropout)
        self.output_norm = nn.LayerNorm(dim)

    def forward(self, graph: RNAGraph, node_states: torch.Tensor) -> torch.Tensor:
        if graph.num_edges() == 0:
            return node_states
        src, dst = graph.edge_index[0], graph.edge_index[1]
        edge_type = graph.edge_type.long()
        normalized = self.pre_norm(node_states)
        src_state = normalized[src]
        dst_state = normalized[dst]
        edge_state = self.edge_embedding(edge_type)
        weights = self.message_weights[edge_type]
        messages = torch.bmm(src_state.unsqueeze(1), weights).squeeze(1)
        gates = torch.sigmoid(self.gate(torch.cat((src_state, dst_state, edge_state), dim=-1)))
        messages = messages * gates
        # Keep reductions in the residual stream dtype (normally FP32). Under
        # BF16 autocast the edge MLP emits BF16 messages, and index_add_ does
        # not perform implicit dtype conversion.
        messages = messages.to(node_states.dtype)
        gates = gates.to(node_states.dtype)
        aggregate = torch.zeros_like(node_states)
        aggregate.index_add_(0, dst, messages)
        degree = torch.zeros(node_states.shape[0], 1, device=node_states.device, dtype=node_states.dtype)
        degree.index_add_(0, dst, torch.ones_like(gates))
        aggregate = aggregate / degree.clamp_min(1.0).sqrt()
        update = self.update(torch.cat((normalized, aggregate), dim=-1))
        return self.output_norm(node_states + self.dropout(update))


class GraphaFold(nn.Module):
    """Sequence-conditioned relational GNN for noncanonical contact detection.

    RiNALMo (or the ablation transformer) supplies global sequence context,
    while the deeper GNN is the main structural encoder over backbone and
    canonical-contact edges.
    """

    def __init__(
        self,
        hidden_dim: int = 384,
        transformer_layers: int = 2,
        transformer_heads: int = 8,
        gnn_layers: int = 8,
        dropout: float = 0.12,
        sequence_backbone: str = "learned",
        rinalmo_model: str = "multimolecule/rinalmo-mega",
        freeze_rinalmo: bool = True,
        max_sequence_length: int = 1024,
    ) -> None:
        super().__init__()
        if hidden_dim % transformer_heads:
            raise ValueError("hidden_dim must be divisible by transformer_heads")
        self.hidden_dim = hidden_dim
        self.sequence_backbone = sequence_backbone
        self.freeze_rinalmo = freeze_rinalmo
        self.max_sequence_length = max_sequence_length
        self.base_embedding = nn.Embedding(5, hidden_dim, padding_idx=4)
        if sequence_backbone == "learned":
            self.position_embedding = nn.Embedding(max_sequence_length, hidden_dim)
            sequence_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=transformer_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.sequence_encoder = nn.TransformerEncoder(
                sequence_layer,
                num_layers=transformer_layers,
                norm=nn.LayerNorm(hidden_dim),
            )
            self.rinalmo_backbone = None
            self.rinalmo_tokenizer = None
            self.rinalmo_projection = None
        elif sequence_backbone == "rinalmo":
            self.position_embedding = None
            try:
                from multimolecule import RiNALMoModel, RnaTokenizer
            except ImportError as exc:
                raise ImportError(
                    "RiNALMo mode requires multimolecule; install requirements-rinalmo.txt"
                ) from exc
            self.sequence_encoder = None
            self.rinalmo_backbone, loading_info = RiNALMoModel.from_pretrained(
                rinalmo_model,
                output_loading_info=True,
            )
            missing_keys = list(loading_info.get("missing_keys", ()))
            model_key_count = max(1, len(self.rinalmo_backbone.state_dict()))
            missing_fraction = len(missing_keys) / model_key_count
            critical_embedding_missing = any(
                key.endswith("embeddings.word_embeddings.weight") for key in missing_keys
            )
            if critical_embedding_missing or missing_fraction > 0.05:
                preview = ", ".join(missing_keys[:8])
                raise RuntimeError(
                    "RiNALMo pretrained weights were not loaded correctly: "
                    f"{len(missing_keys)}/{model_key_count} model keys are missing "
                    f"(first: {preview}). Check multimolecule/transformers versions."
                )
            self.rinalmo_tokenizer = RnaTokenizer.from_pretrained(rinalmo_model)
            rinalmo_dim = int(self.rinalmo_backbone.config.hidden_size)
            self.rinalmo_projection = nn.Sequential(
                nn.LayerNorm(rinalmo_dim),
                nn.Linear(rinalmo_dim, hidden_dim),
                nn.GELU(),
                nn.LayerNorm(hidden_dim),
            )
            if freeze_rinalmo:
                self.rinalmo_backbone.requires_grad_(False)
                self.rinalmo_backbone.eval()
        else:
            raise ValueError("sequence_backbone must be 'learned' or 'rinalmo'")
        self.graph_blocks = nn.ModuleList(
            RelationalGraphBlock(hidden_dim, num_edge_types=2, dropout=dropout)
            for _ in range(gnn_layers)
        )
        self.pair_type_embedding = nn.Embedding(25, 32)
        self.distance_embedding = nn.Embedding(10, 32)
        pair_dim = hidden_dim * 3 + 64
        self.pair_head = nn.Sequential(
            nn.LayerNorm(pair_dim),
            nn.Linear(pair_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def _sequence_context(self, graph: RNAGraph, states: torch.Tensor) -> torch.Tensor:
        lengths = graph.batch_num_nodes().tolist()
        max_length = max(lengths)
        batch_size = len(lengths)
        padded = states.new_zeros((batch_size, max_length, self.hidden_dim))
        padding_mask = torch.ones((batch_size, max_length), dtype=torch.bool, device=states.device)
        offset = 0
        for batch_index, length in enumerate(lengths):
            padded[batch_index, :length] = states[offset : offset + length]
            padding_mask[batch_index, :length] = False
            offset += length
        encoded = self.sequence_encoder(padded, src_key_padding_mask=padding_mask)
        return torch.cat([encoded[i, :length] for i, length in enumerate(lengths)], dim=0)

    def _rinalmo_context(self, graph: RNAGraph) -> torch.Tensor:
        if self.rinalmo_backbone is None or self.rinalmo_tokenizer is None or self.rinalmo_projection is None:
            raise RuntimeError("RiNALMo backbone is not initialized")
        id_to_base = ("A", "C", "G", "U", "N")
        lengths = graph.batch_num_nodes().tolist()
        base_ids = graph.base_id.detach().cpu().tolist()
        sequences: list[str] = []
        offset = 0
        for length in lengths:
            sequences.append("".join(id_to_base[index] for index in base_ids[offset : offset + length]))
            offset += length
        tokenized = self.rinalmo_tokenizer(
            sequences,
            return_tensors="pt",
            padding=True,
            truncation=False,
        )
        input_ids = tokenized["input_ids"].to(graph.base_id.device)
        attention_mask = tokenized["attention_mask"].to(graph.base_id.device)
        context = torch.no_grad() if self.freeze_rinalmo else nullcontext()
        if self.freeze_rinalmo:
            self.rinalmo_backbone.eval()
        with context:
            hidden = self.rinalmo_backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
            ).last_hidden_state
        residues: list[torch.Tensor] = []
        for batch_index, length in enumerate(lengths):
            if length + 2 > hidden.shape[1]:
                raise ValueError(
                    f"RiNALMo tokenized length {hidden.shape[1]} is too short for {length} residues"
                )
            residues.append(hidden[batch_index, 1 : 1 + length])
        return self.rinalmo_projection(torch.cat(residues, dim=0))

    def encode(self, graph: RNAGraph) -> torch.Tensor:
        base_ids = graph.base_id.long()
        base_states = self.base_embedding(base_ids)
        if self.sequence_backbone == "rinalmo":
            states = self._rinalmo_context(graph) + base_states
        else:
            lengths = graph.batch_num_nodes().tolist()
            if max(lengths) > self.max_sequence_length:
                raise ValueError(
                    f"Sequence length {max(lengths)} exceeds learned positional embedding "
                    f"limit {self.max_sequence_length}"
                )
            positions = torch.cat(
                [torch.arange(length, device=base_states.device) for length in lengths]
            )
            states = base_states + self.position_embedding(positions)
            states = self._sequence_context(graph, states)
        for block in self.graph_blocks:
            states = block(graph, states)
        return states

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_rinalmo and self.rinalmo_backbone is not None:
            self.rinalmo_backbone.eval()
        return self

    def score_pairs(self, graph: RNAGraph, states: torch.Tensor, pairs: torch.Tensor) -> torch.Tensor:
        i, j = pairs[:, 0], pairs[:, 1]
        hi, hj = states[i], states[j]
        base_i = graph.base_id[i].long()
        base_j = graph.base_id[j].long()
        lo, hi_base = torch.minimum(base_i, base_j), torch.maximum(base_i, base_j)
        pair_type = lo * 5 + hi_base
        distance = (graph.position[i] - graph.position[j]).abs().long()
        boundaries = torch.tensor([2, 4, 8, 16, 32, 64, 128, 256, 512], device=distance.device)
        distance_bucket = torch.bucketize(distance, boundaries)
        features = torch.cat(
            (
                hi + hj,
                torch.abs(hi - hj),
                hi * hj,
                self.pair_type_embedding(pair_type),
                self.distance_embedding(distance_bucket),
            ),
            dim=-1,
        )
        return self.pair_head(features).squeeze(-1)

    def forward(self, graph: RNAGraph, pairs: torch.Tensor) -> torch.Tensor:
        states = self.encode(graph)
        return self.score_pairs(graph, states, pairs)

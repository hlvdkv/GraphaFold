import unittest

import torch

from graphafold.data import RNAGraph
from graphafold.model import GraphaFold


class LearnedBackboneTests(unittest.TestCase):
    def test_ablation_learns_token_and_position_embeddings_without_rinalmo(self):
        model = GraphaFold(
            hidden_dim=32,
            transformer_layers=1,
            transformer_heads=4,
            gnn_layers=1,
            sequence_backbone="learned",
            max_sequence_length=512,
        )
        self.assertIsNone(model.rinalmo_backbone)
        self.assertEqual(tuple(model.base_embedding.weight.shape), (5, 32))
        self.assertEqual(tuple(model.position_embedding.weight.shape), (512, 32))
        self.assertTrue(model.base_embedding.weight.requires_grad)
        self.assertTrue(model.position_embedding.weight.requires_grad)

        graph = RNAGraph(
            base_id=torch.tensor([0, 2, 1, 3]),
            position=torch.arange(4),
            edge_index=torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]]),
            edge_type=torch.zeros(6, dtype=torch.long),
            node_counts=(4,),
        )
        pairs = torch.tensor([[0, 2], [1, 3]])
        self.assertEqual(tuple(model(graph, pairs).shape), (2,))


if __name__ == "__main__":
    unittest.main()

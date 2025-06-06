from pathlib import Path
import numpy as np

from graphafold.dataset import GraphDataset

class TestGraphDataset:
    dataset_path = Path('tests/dataset_data')

    def test_edge_labels(self):
        """Test that edge labels are correctly assigned."""
        dataset = GraphDataset(self.dataset_path, validation=False)
        sample = dataset[0]
        graph, sequence, edge_candidates, edge_labels = sample
        
        assert edge_labels.shape[0] == edge_candidates.shape[0], \
            "Edge labels and candidates should have the same number of edges."
        
        # Check that edge labels are binary (0 or 1)
        assert set(edge_labels.numpy()) <= {0, 1}, "Edge labels should be binary."
        assert np.sum(edge_labels.numpy()) == len(edge_labels)//2, \
            "In non-validation mode, half of the edges should be positive (1) and half negative (0)."
        
    def test_edge_labels_val(self):
        """Test that edge labels are correctly assigned in validation mode."""
        dataset = GraphDataset(self.dataset_path, validation=True)
        sample = dataset[0]
        graph, sequence, edge_candidates, edge_labels = sample
        
        assert edge_labels.shape[0] == edge_candidates.shape[0], \
            "Edge labels and candidates should have the same number of edges."
        
        # Check that edge labels are binary (0 or 1)
        assert set(edge_labels.numpy()) <= {0, 1}, "Edge labels should be binary."
        
        # In validation mode, all combinations of nodes are considered
        assert np.sum(edge_labels.numpy()) > 0, "There should be positive edges in validation mode."
        assert np.sum(edge_labels.numpy()) < len(edge_labels)//2, \
            "In validation mode, there should be more negative edges than positive ones."
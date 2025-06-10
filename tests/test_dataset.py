from pathlib import Path
import numpy as np
import dgl

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
        
    def test_range_sampling(self):
        """Test that range sampling works correctly."""
        dataset = GraphDataset(self.dataset_path, validation=True, val_sampling_mode="range", val_sampling_range=3)
        num_nodes = 20
        neigh1 = np.arange(10)
        src1, dst1 = neigh1[:-1], neigh1[1:]
        neigh2 = np.arange(10, 20)
        src2, dst2 = neigh2[:-1], neigh2[1:]
        cn_src = [4,5]
        cn_dst = [14, 15]
        
        cn_edges = np.array([cn_src, cn_dst]).T
        expected_sampling_edges = np.array([[1, 11], 
                                            [1, 12],
                                            [1, 13],
                                            [2, 11],
                                            [2, 12],
                                            [2, 13],
                                            [3, 11],
                                            [3, 12],
                                            [3, 13],
                                            [6, 16],
                                            [6, 17],
                                            [6, 18],
                                            [7, 16],
                                            [7, 17],
                                            [7, 18],
                                            [8, 16],
                                            [8, 17],
                                            [8, 18],
                                            [1, 16],
                                            [1, 17],
                                            # [1, 18], # extreme values in range
                                            [2, 16],
                                            [2, 17],
                                            [2, 18],
                                            [3, 16],
                                            [3, 17],
                                            [3, 18],
                                            [6, 11],
                                            [6, 12],
                                            [6, 13],
                                            [7, 11],
                                            [7, 12],
                                            [7, 13],
                                            # [8, 11], # extreme values in range
                                            [8, 12],
                                            [8, 13]])
        sampling_edges = dataset.get_range_sampling(cn_edges, num_nodes)
        # sort edges by 1 column, then by 2 column
        sam_edges_set = set(map(tuple, sampling_edges.tolist()))
        expected_sampling_edges_set = set(map(tuple, expected_sampling_edges.tolist()))
        assert sam_edges_set == expected_sampling_edges_set, \
            "Range sampling did not produce the expected edges."



        
        
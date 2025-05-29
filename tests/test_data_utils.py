from pathlib import Path
import numpy as np
from graphafold.data import load_idx_file
from graphafold.data import Sample

class TestUtils:
    test_idx = Path(__file__).parent / 'data' / 'test.idx'
    
    def test_load_idx(self):
        """Test loading of idx file."""
        expected_neighbors = [
            (0, 1), (1, 2), (2, 3), (4, 5),
            (5, 6), (6, 7), (8, 9), (9, 10),
            (10, 11), (12, 13), (13, 14)
        ]
        index_to_nt, neighbors = load_idx_file(self.test_idx)
        assert len(index_to_nt) == 15
        assert len(neighbors) == 11
        assert neighbors == expected_neighbors
    
    def test_sample(self):
        """Test Sample class initialization."""
        expected_sequences = ['GCGU', 'ACAG', 'GGAA', 'CAC']
        index_to_nt, neighbours = load_idx_file(self.test_idx)
        sample = Sample(np.zeros((15,15)), index_to_nt, neighbours)
        
        assert sample.sequences == expected_sequences

    def test_sequence_breaks(self):
        """Test sequence breaks extraction."""
        index_to_nt, neighbours = load_idx_file(self.test_idx)
        sample = Sample(np.zeros((15,15)), index_to_nt, neighbours)
        
        expected_breaks = [4, 9, 14]
        assert sample.sequnce_breaks == expected_breaks


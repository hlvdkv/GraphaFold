import pickle
import numpy as np


class Sample():
    
    def __init__(self, matrix:np.ndarray, index_nt_dict:dict, neighbours:list[tuple[int, int]]):
        self.matrix = np.array(matrix)
        self.num_nodes = matrix.shape[0]
        self.cn = np.where(matrix == 1)
        self.non_cn = np.where(matrix > 1)
        self.index_nt_dict = index_nt_dict
        self.neighbours = neighbours # edges a-b and b-a
        a_b_neigbours = neighbours[:len(neighbours)//2]
        self.sequences = self.extract_contiguous_sequences(index_nt_dict, a_b_neigbours)
        self.sequnce_breaks = self.get_sequence_breaks(self.sequences)
        pass

    def extract_contiguous_sequences(self, index_to_nt, neighbors):
        """
        Convert neighbours and index_nt_dict to a sequences.
        """
        sequences = []
        current_sequence = []

        for i, (a, b) in enumerate(neighbors):
            if not current_sequence:
                current_sequence.append(index_to_nt[a+1])
            
            current_sequence.append(index_to_nt[b+1])

            # Check if next neighbor is not consecutive
            if i + 1 < len(neighbors):
                next_a, next_b = neighbors[i + 1]
                if next_a != b:
                    sequences.append(''.join(current_sequence))
                    current_sequence = []

        # Add any remaining sequence
        if current_sequence:
            sequences.append(''.join(current_sequence))

        return sequences

    def get_sequence_breaks(self, sequences, sep_len=1):
        """
        Get the breaks in the sequences.
        """
        break_indices = []
        current_index = 0

        for seq in sequences[:-1]:  # skip the last, no break after it
            current_index += len(seq)
            break_indices.append(current_index)
            current_index +=  sep_len # account for separator length

        return break_indices
    
    def is_valid(self):
        """ Check if the sample is valid."""
        if np.sum(self.cn) == 0:
            print("Sample is invalid: no canonical edges found.")
            return False
        if np.sum(self.non_cn) == 0:
            print("Sample is invalid: no non-canonical edges found.")
            return False
        if len(self.index_nt_dict) != self.num_nodes:
            print(f"Sample is invalid: index_nt_dict length {len(self.index_nt_dict)} does not match num_nodes {self.num_nodes}.")
            return False
        if len("".join(self.sequences)) != len(self.matrix):
            print(f"Sample is invalid: total sequence length {len(''.join(self.sequences))} does not match matrix size {len(self.matrix)}.")
            return False
        if len(self.neighbours) == 0:
            print("Sample is invalid: no neighbours found.")
            return False

        return True
    
    def save_to_pickle(self, file_path):
        """
        Save the sample to a pickle file.
        """
        with open(file_path, 'wb') as f:
            pickle.dump(self, f)
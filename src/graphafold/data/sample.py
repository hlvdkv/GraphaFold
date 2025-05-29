import pickle
import numpy as np


class Sample():
    
    def __init__(self, matrix:np.ndarray, index_nt_dict:dict, neighbours:list[tuple[int, int]]):
        self.matrix = np.array(matrix)
        self.num_nodes = matrix.shape[0]
        self.cn = np.column_stack(np.where(matrix == 1)) # size [N,2]
        self.non_cn = np.column_stack(np.where(matrix > 1)) # size [N,2]
        self.index_nt_dict = index_nt_dict
        self.neighbours = neighbours # TODO: add the reversed columns A-B and B-A
        self.sequences = self.extract_contiguous_sequences(index_nt_dict, neighbours)
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
    
    def save_to_pickle(self, file_path):
        """
        Save the sample to a pickle file.
        """
        with open(file_path, 'wb') as f:
            pickle.dump(self, f)
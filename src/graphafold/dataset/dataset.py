import os
import pickle
import dgl
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from graphafold.data import Sample

class GraphDataset(Dataset):
    def __init__(self, path, validation:bool=False, sequence_sep:str="-"):
        super(GraphDataset, self).__init__()
        self.path = path
        self.samples = os.listdir(path)
        self.graphs = []
        self.edge_lists = []
        self.labels_list = []
        self.file_names = []
        self.validation = validation
        self.sequence_sep = sequence_sep

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        sample_path = os.path.join(self.path, sample)
        with open(sample_path, 'rb') as f:
            data = pickle.load(f)
         # Add canonical and neighbor edges
        src_cn, dst_cn = data.cn
        src_nb, dst_nb = data.neighbours[:, 0], data.neighbours[:, 1]
        all_src = np.concatenate([src_cn, src_nb])
        all_dst = np.concatenate([dst_cn, dst_nb])
        graph = dgl.graph((all_src, all_dst), num_nodes=data.num_nodes)

        # Assign edge features
        num_cn = len(src_cn)
        num_nb = len(src_nb)
        edge_features = torch.zeros(num_cn + num_nb, 2)
        edge_features[:num_cn, 0] = 1  # canonical: [1, 0]
        edge_features[num_cn:, 1] = 1  # neighbor: [0, 1]
        graph.edata['feat'] = edge_features
        sequence = self.sequence_sep.join(data.sequences)

        edge_candidates, edge_labels = self.get_edge_candidates(data)
        assert len(edge_candidates) >0, f"Edge candidates should not be empty in sample {sample_path}"

        return graph, sequence, edge_candidates, edge_labels
    
    def get_edge_candidates(self, sample:Sample):
        """
        Generate edge candidates for the graph.
        If self.validation is True, all combinations of nodes are considered (excluding canonical edges).
        If self.validation is False, then all nodes candidates from non-canonical list are included and random sample
        of canonical nodes is added of the same size. Thus, 50% of edges is positive and 50% is negative. However, random
        sample does not include canonical and non-canonical edges that are already present in the graph.
        Positive edges are the non-canonical ones that are to be predicted, while negative edges are randomly sampled (ones
        that do not exist in the graph). Canonical edges are not included in the candidates and for this reason they are ignored.
        Returns:
            edge_candidates (torch.Tensor): Tensor of shape [K, 2] where K is the number of edge candidates.
            edge_labels (torch.Tensor): Tensor of shape [K] with labels for each edge candidate (1 for positive, 0 for negative).
        """
        num_nodes = sample.num_nodes
        # Canonical edges (to ignore)
        cn_edges = set((int(i), int(j)) for i, j in zip(sample.cn[0], sample.cn[1]) if int(i) < int(j))
        # Non-canonical edges (positives)
        non_cn_edges = set((int(i), int(j)) for i, j in zip(sample.non_cn[0], sample.non_cn[1]) if int(i) < int(j))

        if self.validation:
            # All possible pairs (i, j) with i < j, excluding canonical edges
            all_pairs = set((i, j) for i in range(num_nodes) for j in range(i+1, num_nodes))
            candidate_edges = list(all_pairs - cn_edges)
            # Label as 1 if in non-canonical, else 0
            edge_labels = [1 if edge in non_cn_edges else 0 for edge in candidate_edges]
        else:
            # Positive: all non-canonical edges
            pos_edges = list(non_cn_edges)
            num_pos = len(pos_edges)
            # All possible pairs (i, j) with i < j, excluding canonical and non-canonical edges
            all_pairs = set((i, j) for i in range(num_nodes) for j in range(i+1, num_nodes))
            forbidden = cn_edges | non_cn_edges
            neg_pool = list(all_pairs - forbidden)
            # Randomly sample negatives, same number as positives
            if len(neg_pool) >= num_pos:
                neg_edges = random.sample(neg_pool, num_pos)
                # neg_edges = neg_pool[:num_pos]  # take first num_pos from the pool
            else:
                neg_edges = neg_pool  # fallback: use all available
            candidate_edges = pos_edges + neg_edges
            edge_labels = [1] * len(pos_edges) + [0] * len(neg_edges)

        edge_candidates = torch.tensor(candidate_edges, dtype=torch.long)
        edge_labels = torch.tensor(edge_labels, dtype=torch.float)
        return edge_candidates, edge_labels



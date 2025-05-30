import os
import dgl
import numpy as np
import torch
from graphafold.const import nucleotide_to_onehot

def load_matrix(file):
    return np.loadtxt(file, delimiter=',')

def load_idx_file(idx_file_path):
    with open(idx_file_path, 'r') as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines if line.strip() and not line.startswith('#')]
    index_to_nt = {}
    neighbors = []

    parsed = []

    for line in lines:
        index_str, rna = line.strip().split(',')
        index = int(index_str)
        chain, rest = rna.split('.')
        nt = rest[0]
        res_id = int(rest[1:])
        parsed.append((index, chain, nt, res_id))
        index_to_nt[index] = nt

    for i in range(1, len(parsed)):
        idx1, chain1, _, res1 = parsed[i - 1]
        idx2, chain2, _, res2 = parsed[i]
        if chain1 == chain2 and res2 - res1 == 1:
            neighbors.append((idx1-1, idx2-1))

    return index_to_nt, np.array(neighbors)

def custom_collate(batch):
    graphs, labels1, labels2 = zip(*batch)
    batched_graph = dgl.batch(graphs)

    # You can return labels as-is (list of lists) or process them here
    return batched_graph, list(labels1), list(labels2)

def pad_matrix(matrix, num_nodes):
    current_size = matrix.shape[0]
    if current_size >= num_nodes:
        return matrix
    else:
        padded_matrix = np.zeros((num_nodes, num_nodes))
        padded_matrix[:current_size, :current_size] = matrix
        return padded_matrix

def cmt2graph(cmt_file_path, idx_file_path, graphs=[], edge_lists=[], labels_list=[], file_names=[]):
    if not os.path.exists(cmt_file_path) or not os.path.exists(idx_file_path):
        raise FileNotFoundError(f"File not found: {cmt_file_path} or {idx_file_path}")

    cmt_matrix = load_matrix(cmt_file_path)

    noncanonical_matrix = cmt_matrix # np.zeros_like(cmt_matrix)
    neighborhood_matrix = np.where(cmt_matrix == -1, 1, 0) # np.zeros_like(cmt_matrix)

    node_nucleotides_dict, max_index_idx = load_idx_file(idx_file_path)
    non_zero_rows = np.where(np.sum(np.abs(cmt_matrix), axis=1) > 0)[0]
    non_zero_cols = np.where(np.sum(np.abs(cmt_matrix), axis=0) > 0)[0]
    if len(non_zero_rows) == 0 and len(non_zero_cols) == 0:
        raise ValueError(f"Empty CMT matrix: {cmt_file_path}")
    
    max_index_cmt = max(np.max(non_zero_rows), np.max(non_zero_cols))
    num_nodes = max(max_index_cmt, max_index_idx) + 1

    cmt_matrix = pad_matrix(cmt_matrix, num_nodes)
    noncanonical_matrix = pad_matrix(noncanonical_matrix, num_nodes)
    neighborhood_matrix = pad_matrix(neighborhood_matrix, num_nodes)

    src, dst = np.where(cmt_matrix == 1)
    g = dgl.graph((src, dst), num_nodes=num_nodes)
    neigh_src, neigh_dst = np.where(neighborhood_matrix == 1)
    g.add_edges(neigh_src, neigh_dst)

    feature_size = 4
    node_features = torch.zeros(num_nodes, feature_size)
    for i in range(num_nodes):
        nucleotide = node_nucleotides_dict.get(i, None)
        onehot = nucleotide_to_onehot.get(nucleotide, [0, 0, 0, 0])
        node_features[i, :] = torch.tensor(onehot)
    g.ndata['feat'] = node_features

    num_edges = g.number_of_edges()
    edge_features = torch.zeros(num_edges, 2)
    u, v = g.edges()
    u = u.numpy()
    v = v.numpy()
    for idx_edge, (i, j) in enumerate(zip(u, v)):
        if noncanonical_matrix[i, j] == 1:
            edge_features[idx_edge] = torch.tensor([1, 1])
        elif cmt_matrix[i, j] == 1:
            edge_features[idx_edge] = torch.tensor([1, 0])
        else:
            edge_features[idx_edge] = torch.tensor([0, 1])
    g.edata['feat'] = edge_features

    positive_indices = []
    negative_indices = []
    for i in range(num_nodes):
        for j in range(num_nodes):
            if noncanonical_matrix[i, j] == 1:
                positive_indices.append((i, j))
            else:
                nucleotide_i = node_nucleotides_dict.get(i, None)
                nucleotide_j = node_nucleotides_dict.get(j, None)
                if nucleotide_i is None or nucleotide_j is None:
                    continue
                negative_indices.append((i, j))

    positive_edges = torch.tensor(positive_indices, dtype=torch.long) if len(positive_indices) > 0 else torch.empty((0, 2), dtype=torch.long)
    negative_edges = torch.tensor(negative_indices, dtype=torch.long) if len(negative_indices) > 0 else torch.empty((0, 2), dtype=torch.long)

    # if len(positive_edges) == 0 or len(negative_edges) == 0:
        # raise ValueError(f"No positive or negative edges found in the CMT matrix: {cmt_file_path}")


    if len(negative_edges) >= len(positive_edges):
        balanced_negative_indices = torch.randperm(len(negative_edges))[:len(positive_edges)]
        balanced_negative_edges = negative_edges[balanced_negative_indices]
    else:
        balanced_negative_edges = negative_edges

    combined_edges = torch.cat([positive_edges, balanced_negative_edges], dim=0)
    edge_labels = torch.cat([
        torch.ones(len(positive_edges)),
        torch.zeros(len(balanced_negative_edges))
    ]).long()

    graphs.append(g)
    edge_lists.append(combined_edges)
    labels_list.append(edge_labels)
    file_names.append(cmt_file_path)
    return graphs, edge_lists, labels_list, file_names
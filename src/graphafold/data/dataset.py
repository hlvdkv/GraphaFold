import dgl
import torch
from torch.utils.data import Dataset

class GraphDataset(Dataset):
    def __init__(self, graphs, edge_lists, labels_list, file_names):
        self.graphs = graphs
        self.edge_lists = edge_lists
        self.labels_list = labels_list
        self.file_names = file_names

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx], self.edge_lists[idx], self.labels_list[idx], self.file_names[idx]

def collate(samples):
    graphs, edge_lists, labels, file_names = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    node_id_offsets = []
    offset = 0
    for g in graphs:
        node_id_offsets.append(offset)
        offset += g.number_of_nodes()
    adjusted_edge_lists = []
    adjusted_labels = []
    for edge_list, label, offset in zip(edge_lists, labels, node_id_offsets):
        adjusted_edges = edge_list + offset
        adjusted_edge_lists.append(adjusted_edges)
        adjusted_labels.append(label)
    combined_edge_list = torch.cat(adjusted_edge_lists, dim=0)
    combined_labels = torch.cat(adjusted_labels, dim=0)
    return batched_graph, combined_edge_list, combined_labels, file_names
import os
import numpy as np
import torch
import dgl
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
import pandas as pd
import matplotlib.pyplot as plt 

def load_matrix(file):
    return np.loadtxt(file, delimiter=',')

def load_idx_file(idx_file_path):
    node_nucleotides_dict = {}
    indices = []
    with open(idx_file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(',')
            if len(parts) != 2:
                continue
            index_str, code_str = parts
            index = int(index_str.strip()) - 1 
            code = code_str.strip()
            # Extract nucleotide
            if '.' in code:
                _, nucleotide_code = code.split('.')
                nucleotide = nucleotide_code[0]  # First character after the dot
                node_nucleotides_dict[index] = nucleotide
            else:
                node_nucleotides_dict[index] = None
            indices.append(index)
    max_index = max(indices) if indices else -1
    return node_nucleotides_dict, max_index

def pad_matrix(matrix, num_nodes):
    current_size = matrix.shape[0]
    if current_size >= num_nodes:
        return matrix
    else:
        padded_matrix = np.zeros((num_nodes, num_nodes))
        padded_matrix[:current_size, :current_size] = matrix
        return padded_matrix

train_folder = 'after2/amt'
cmt_folder = 'after2/cmt'
idx_folder = 'after2/idx'  

graphs = []
edge_lists = []
labels_list = []
file_names = []  

for amt_file in os.listdir(train_folder):
    if amt_file.endswith('.amt'):
        cmt_file = amt_file.replace('.amt', '.cmt')
        idx_file = amt_file.replace('.amt', '.idx')
        cmt_file_path = os.path.join(cmt_folder, cmt_file)
        idx_file_path = os.path.join(idx_folder, idx_file)
        if not os.path.exists(cmt_file_path):
            print(f'Missing file: {cmt_file_path}')
            continue
        if not os.path.exists(idx_file_path):
            print(f'Missing file: {idx_file_path}')
            continue

        cmt_matrix = load_matrix(cmt_file_path)
        amt_matrix = load_matrix(os.path.join(train_folder, amt_file))

        # Non-canonical edges: values > 1
        noncanonical_matrix = np.where(amt_matrix > 1, 1, 0)
        neighborhood_matrix = np.where(amt_matrix == -1, 1, 0)

        node_nucleotides_dict, max_index_idx = load_idx_file(idx_file_path)

        # Determine actual number of nodes
        non_zero_rows = np.where(np.sum(np.abs(cmt_matrix), axis=1) > 0)[0]
        non_zero_cols = np.where(np.sum(np.abs(cmt_matrix), axis=0) > 0)[0]
        if len(non_zero_rows) == 0 and len(non_zero_cols) == 0:
            print(f"Matrix {cmt_file_path} is empty.")
            continue
        max_index_cmt = max(np.max(non_zero_rows), np.max(non_zero_cols))

        num_nodes = max(max_index_cmt, max_index_idx) + 1

        cmt_matrix = pad_matrix(cmt_matrix, num_nodes)
        amt_matrix = pad_matrix(amt_matrix, num_nodes)
        noncanonical_matrix = pad_matrix(noncanonical_matrix, num_nodes)
        neighborhood_matrix = pad_matrix(neighborhood_matrix, num_nodes)

        # Canonical edges
        src, dst = np.where(cmt_matrix == 1)
        graph = dgl.graph((src, dst), num_nodes=num_nodes)

        # Add neighborhood edges
        neigh_src, neigh_dst = np.where(neighborhood_matrix == 1)
        graph.add_edges(neigh_src, neigh_dst)

        # Nucleotide one-hot encoding
        nucleotide_to_onehot = {
            'A': [1, 0, 0, 0],
            'C': [0, 1, 0, 0],
            'G': [0, 0, 1, 0],
            'U': [0, 0, 0, 1],
            None: [0, 0, 0, 0] 
        }

        # Node features
        feature_size = 4  # one-hot encoding of nucleotides
        node_features = torch.zeros(num_nodes, feature_size)
        for i in range(num_nodes):
            nucleotide = node_nucleotides_dict.get(i, None)
            onehot = nucleotide_to_onehot.get(nucleotide, [0, 0, 0, 0])
            node_features[i, :] = torch.tensor(onehot)
        graph.ndata['feat'] = node_features

        num_edges = graph.number_of_edges()
        edge_features = torch.zeros(num_edges, 2)
        u, v = graph.edges()
        u = u.numpy()
        v = v.numpy()
        for idx_edge, (i, j) in enumerate(zip(u, v)):
            if noncanonical_matrix[i, j] == 1:
                edge_features[idx_edge] = torch.tensor([1, 1])
            elif cmt_matrix[i, j] == 1:
                edge_features[idx_edge] = torch.tensor([1, 0])
            else:
                edge_features[idx_edge] = torch.tensor([0, 1])
        graph.edata['feat'] = edge_features

        positive_indices = [(i, j) for i in range(num_nodes) for j in range(num_nodes) if noncanonical_matrix[i, j] == 1]

        # Negative sampling
        num_neg_samples_per_node = 5
        negative_indices = []
        for i in range(num_nodes):
            positive_neighbors = {j for (u, j) in positive_indices if u == i}
            canonical_neighbors = set(np.where(cmt_matrix[i] == 1)[0])
            excluded = positive_neighbors.union(canonical_neighbors).union({i})
            possible_negatives = list(set(range(num_nodes)) - excluded)
            if possible_negatives:
                sample_size = min(num_neg_samples_per_node, len(possible_negatives))
                sampled = np.random.choice(possible_negatives, sample_size, replace=False)
                negative_indices.extend([(i, j) for j in sampled])

        if len(positive_indices) == 0:
            continue
        if len(negative_indices) == 0:
            continue

        # Convert to tensors
        positive_edges = torch.tensor(positive_indices, dtype=torch.long)
        negative_edges = torch.tensor(negative_indices, dtype=torch.long)

        # Balance the dataset
        if len(negative_edges) >= len(positive_edges):
            perm = torch.randperm(len(negative_edges))
            balanced_negative_edges = negative_edges[perm[:len(positive_edges)]]
        else:
            balanced_negative_edges = negative_edges

        # Combine edges and labels
        combined_edges = torch.cat([positive_edges, balanced_negative_edges], dim=0)
        edge_labels = torch.cat([
            torch.ones(len(positive_edges)),
            torch.zeros(len(balanced_negative_edges))
        ]).long()

        graphs.append(graph)
        edge_lists.append(combined_edges)
        labels_list.append(edge_labels)
        file_names.append(amt_file)  

# Dataset definition
class GraphDataset(torch.utils.data.Dataset):
    def __init__(self, graphs, edge_lists, labels_list, file_names):
        self.graphs = graphs
        self.edge_lists = edge_lists
        self.labels_list = labels_list
        self.file_names = file_names

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx], self.edge_lists[idx], self.labels_list[idx], self.file_names[idx]

# Collate function for batching
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

# Create dataset and split
dataset = GraphDataset(graphs, edge_lists, labels_list, file_names)
indices = list(range(len(dataset)))
train_indices, temp_indices = train_test_split(indices, test_size=0.3, random_state=42)
val_indices, test_indices = train_test_split(temp_indices, test_size=0.5, random_state=42)
train_dataset = torch.utils.data.Subset(dataset, train_indices)
val_dataset = torch.utils.data.Subset(dataset, val_indices)
test_dataset = torch.utils.data.Subset(dataset, test_indices)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate)

# GNN model
from dgl.nn import NNConv

class GNNModel(nn.Module):
    def __init__(self, in_feats, edge_feats, hidden_feats, out_feats):
        super(GNNModel, self).__init__()
        self.node_proj = nn.Linear(in_feats, hidden_feats)
        edge_nn = nn.Sequential(
            nn.Linear(edge_feats, hidden_feats * hidden_feats),
            nn.ReLU()
        )
        self.conv1 = NNConv(hidden_feats, hidden_feats, edge_nn, aggregator_type='mean')
        self.conv2 = NNConv(hidden_feats, hidden_feats, edge_nn, aggregator_type='mean')
        self.edge_predictor = nn.Sequential(
            nn.Linear(hidden_feats * 2, hidden_feats),
            nn.ReLU(),
            nn.Linear(hidden_feats, out_feats)
        )

    def forward(self, g, edge_list):
        h = g.ndata['feat']
        e = g.edata['feat']
        h = self.node_proj(h)
        h = self.conv1(g, h, e)
        h = F.relu(h)
        h = self.conv2(g, h, e)
        h = F.relu(h)
        src_nodes = edge_list[:, 0]
        dst_nodes = edge_list[:, 1]
        src_h = h[src_nodes]
        dst_h = h[dst_nodes]
        edge_inputs = torch.cat([src_h, dst_h], dim=1)
        logits = self.edge_predictor(edge_inputs)
        return logits

# Model initialization
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GNNModel(in_feats=4, edge_feats=2, hidden_feats=16, out_feats=2)
model.to(device)

optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

def calculate_metrics(logits, labels):
    _, predicted = torch.max(logits, 1)
    labels = labels.cpu().numpy()
    predicted = predicted.cpu().numpy()
    if len(np.unique(labels)) == 1:
        precision = recall = f1 = accuracy = 1.0 if np.all(labels == predicted) else 0.0
    else:
        precision = precision_score(labels, predicted, zero_division=0)
        recall = recall_score(labels, predicted, zero_division=0)
        f1 = f1_score(labels, predicted, zero_division=0)
        accuracy = np.mean(predicted == labels)
    return accuracy, precision, recall, f1

# Training loop
num_epochs = 80
best_val_loss = float('inf')
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    epoch_accuracy = 0
    for batched_graph, combined_edge_list, combined_labels, _ in train_loader:
        batched_graph = batched_graph.to(device)
        combined_edge_list = combined_edge_list.to(device)
        combined_labels = combined_labels.to(device)
        logits = model(batched_graph, combined_edge_list)
        loss = loss_fn(logits, combined_labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        acc, _, _, _ = calculate_metrics(logits, combined_labels)
        epoch_accuracy += acc
    epoch_loss /= len(train_loader)
    epoch_accuracy /= len(train_loader)
    train_losses.append(epoch_loss)
    train_accuracies.append(epoch_accuracy)
    
    model.eval()
    val_loss = 0
    val_accuracy = 0
    with torch.no_grad():
        for batched_graph, combined_edge_list, combined_labels, _ in val_loader:
            batched_graph = batched_graph.to(device)
            combined_edge_list = combined_edge_list.to(device)
            combined_labels = combined_labels.to(device)
            logits = model(batched_graph, combined_edge_list)
            loss = loss_fn(logits, combined_labels)
            val_loss += loss.item()
            acc, _, _, _ = calculate_metrics(logits, combined_labels)
            val_accuracy += acc
    val_loss /= len(val_loader)
    val_accuracy /= len(val_loader)
    val_losses.append(val_loss)
    val_accuracies.append(val_accuracy)
    
    print(f'Epoch {epoch+1}, Training loss: {epoch_loss:.4f}, Validation loss: {val_loss:.4f}')
    print(f'Training accuracy: {epoch_accuracy:.4f}, Validation accuracy: {val_accuracy:.4f}')
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'model_v4.pth')

print("Training complete")

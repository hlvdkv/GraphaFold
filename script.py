import os
import numpy as np
import torch
import dgl
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import pandas as pd
import sys


helix_output_folder = '/results/helix_outputs'
os.makedirs(helix_output_folder, exist_ok=True)

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
            index = int(index_str.strip()) - 1  # Indeksy zaczynamy od 0
            code = code_str.strip()
            if '.' in code:
                _, nucleotide_code = code.split('.')
                nucleotide = nucleotide_code[0]  # Pierwszy znak po kropce
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


# Definicja datasetu i funkcja collate

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
    # Obliczenie przesunięcia indeksów węzłów
    node_id_offsets = []
    offset = 0
    for g in graphs:
        node_id_offsets.append(offset)
        offset += g.number_of_nodes()
    # Dostosowanie edge_lists – przesunięcie indeksów
    adjusted_edge_lists = []
    adjusted_labels = []
    for edge_list, label, offset in zip(edge_lists, labels, node_id_offsets):
        adjusted_edges = edge_list + offset  # Przesunięcie indeksów
        adjusted_edge_lists.append(adjusted_edges)
        adjusted_labels.append(label)
    combined_edge_list = torch.cat(adjusted_edge_lists, dim=0)
    combined_labels = torch.cat(adjusted_labels, dim=0)
    return batched_graph, combined_edge_list, combined_labels, file_names


if len(sys.argv) < 2:
    print("Usage: python script.py /path/to/folder")
    sys.exit(1)
base_folder = sys.argv[1]

new_amt_folder = os.path.join(base_folder, 'amt')
new_cmt_folder = os.path.join(base_folder, 'cmt')
new_idx_folder = os.path.join(base_folder, 'idx')

graphs = []
edge_lists = []
labels_list = []
file_names = []

# Mapowanie nukleotydów
nucleotide_to_onehot = {
    'A': [1, 0, 0, 0],
    'C': [0, 1, 0, 0],
    'G': [0, 0, 1, 0],
    'U': [0, 0, 0, 1],
    None: [0, 0, 0, 0]
}

for amt_file in os.listdir(new_amt_folder):
    if amt_file.endswith('.amt'):
        cmt_file = amt_file.replace('.amt', '.cmt')
        idx_file = amt_file.replace('.amt', '.idx')
        cmt_file_path = os.path.join(new_cmt_folder, cmt_file)
        idx_file_path = os.path.join(new_idx_folder, idx_file)
        if not os.path.exists(cmt_file_path) or not os.path.exists(idx_file_path):
            print(f'Brak pliku dla {amt_file}')
            continue

        # Wczytanie macierzy
        cmt_matrix = load_matrix(cmt_file_path)
        amt_matrix = load_matrix(os.path.join(new_amt_folder, amt_file))

        noncanonical_matrix = np.where(amt_matrix > 1, 1, 0)
        neighborhood_matrix = np.where(amt_matrix == -1, 1, 0)

        node_nucleotides_dict, max_index_idx = load_idx_file(idx_file_path)

        # Ustalenie liczby węzłów na podstawie macierzy
        non_zero_rows = np.where(np.sum(np.abs(cmt_matrix), axis=1) > 0)[0]
        non_zero_cols = np.where(np.sum(np.abs(cmt_matrix), axis=0) > 0)[0]
        if len(non_zero_rows) == 0 and len(non_zero_cols) == 0:
            print(f"Macierz {cmt_file_path} jest pusta.")
            continue
        max_index_cmt = max(np.max(non_zero_rows), np.max(non_zero_cols))
        num_nodes = max(max_index_cmt, max_index_idx) + 1

        cmt_matrix = pad_matrix(cmt_matrix, num_nodes)
        amt_matrix = pad_matrix(amt_matrix, num_nodes)
        noncanonical_matrix = pad_matrix(noncanonical_matrix, num_nodes)
        neighborhood_matrix = pad_matrix(neighborhood_matrix, num_nodes)

        # Utworzenie grafu – krawędzie kanoniczne
        src, dst = np.where(cmt_matrix == 1)
        g = dgl.graph((src, dst), num_nodes=num_nodes)
        # Dodanie krawędzi sąsiedztwa
        neigh_src, neigh_dst = np.where(neighborhood_matrix == 1)
        g.add_edges(neigh_src, neigh_dst)

        # Inicjalizacja cech węzłów (
        feature_size = 4
        node_features = torch.zeros(num_nodes, feature_size)
        for i in range(num_nodes):
            nucleotide = node_nucleotides_dict.get(i, None)
            onehot = nucleotide_to_onehot.get(nucleotide, [0, 0, 0, 0])
            node_features[i, :] = torch.tensor(onehot)
        g.ndata['feat'] = node_features

        # Inicjalizacja cech krawędzi
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
        positive_edges = torch.tensor(positive_indices, dtype=torch.long)
        negative_edges = torch.tensor(negative_indices, dtype=torch.long)
        if len(positive_edges) == 0 or len(negative_edges) == 0:
            continue
        # Zbalansowanie
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
        file_names.append(amt_file)

if len(graphs) == 0:
    raise ValueError("Brak danych do testowania!")

dataset = GraphDataset(graphs, edge_lists, labels_list, file_names)
test_loader = DataLoader(dataset, batch_size=1, collate_fn=collate)


# Definicja modelu

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
        # Predykcja dla krawędzi
        src_nodes = edge_list[:, 0]
        dst_nodes = edge_list[:, 1]
        src_h = h[src_nodes]
        dst_h = h[dst_nodes]
        edge_inputs = torch.cat([src_h, dst_h], dim=1)
        logits = self.edge_predictor(edge_inputs)
        return logits

# Wczytanie modelu

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GNNModel(in_feats=4, edge_feats=2, hidden_feats=16, out_feats=2)
model.load_state_dict(torch.load('model_v4.pth', map_location=device))
model.to(device)
model.eval()
print("Model został wczytany i jest gotowy do testowania.")


# Funkcja do obliczania metryk 
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

# Testowanie modelu, obliczanie metryk oraz generowanie plików helix
results = []
all_accuracy = []
all_precision = []
all_recall = []
all_f1 = []

with torch.no_grad():
    for batched_graph, combined_edge_list, combined_labels, file_names_batch in test_loader:
        batched_graph = batched_graph.to(device)
        combined_edge_list = combined_edge_list.to(device)
        combined_labels = combined_labels.to(device)

        logits = model(batched_graph, combined_edge_list)
        _, predicted = torch.max(logits, 1)

        # Obliczenie metryk dla batcha
        accuracy, precision, recall, f1 = calculate_metrics(logits, combined_labels)
        all_accuracy.append(accuracy)
        all_precision.append(precision)
        all_recall.append(recall)
        all_f1.append(f1)

        # Rozdzielenie batched_graph na pojedyncze grafy
        graphs_batch = dgl.unbatch(batched_graph)
        num_nodes_list = [g.number_of_nodes() for g in graphs_batch]
        cum_nodes = np.cumsum([0] + num_nodes_list)

       
        for i, graph in enumerate(graphs_batch):
            start = cum_nodes[i]
            end = cum_nodes[i+1]
            mask = (combined_edge_list[:, 0] >= start) & (combined_edge_list[:, 0] < end)
            edge_list = combined_edge_list[mask] - start  # przesunięcie indeksów
            true_labels = combined_labels[mask]
            predicted_labels = predicted[mask]

            # Konwersja krawędzi do postaci krotek
            gt_edges = [tuple(edge.cpu().numpy()) for edge, label in zip(edge_list, true_labels) if label == 1]
            pred_edges = [tuple(edge.cpu().numpy()) for edge, label in zip(edge_list, predicted_labels) if label == 1]

            # Obliczenie metryk per graf
            tp = len(set(gt_edges) & set(pred_edges))
            fn = len(set(gt_edges) - set(pred_edges))
            fp = len(set(pred_edges) - set(gt_edges))
            ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
            inf_metric = np.sqrt(ppv * tpr) if (ppv * tpr) > 0 else 0

            results.append({
                'Nazwa pliku': file_names_batch[i],
                'TP': tp,
                'FN': fn,
                'FP': fp,
                'PPV': ppv,
                'TPR': tpr,
                'INF': inf_metric
            })

            # Przygotowanie danych w formacie helix
            helix_data = []
            for edge in gt_edges:
                if edge in pred_edges:
                    helix_data.append(f"{edge[0]},{edge[1]},PredictedGoodNonCanonical,1.0")
            for edge in set(pred_edges) - set(gt_edges):
                helix_data.append(f"{edge[0]},{edge[1]},PredictedBadNonCanonical,1.0")
            for edge in set(gt_edges) - set(pred_edges):
                helix_data.append(f"{edge[0]},{edge[1]},NotPredictedNonCanonical,1.0")

            helix_file_path = os.path.join(helix_output_folder, f"{file_names_batch[i]}.helix")
            with open(helix_file_path, 'w') as f:
                f.write("\n".join(helix_data) + "\n")

        print("Pliki:", file_names_batch)
        print(f"Dokładność: {accuracy:.4f}, Precyzja: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}\n")


mean_accuracy = np.mean(all_accuracy)
mean_precision = np.mean(all_precision)
mean_recall = np.mean(all_recall)
mean_f1 = np.mean(all_f1)

print("Średnia dokładność: {:.4f}".format(mean_accuracy))
print("Średnia precyzja: {:.4f}".format(mean_precision))
print("Średni recall: {:.4f}".format(mean_recall))
print("Średni F1: {:.4f}".format(mean_f1))


summary = {
    'Nazwa pliku': 'Średnio',
    'TP': '-',
    'FN': '-',
    'FP': '-',
    'PPV': mean_precision,
    'TPR': mean_recall,
    'INF': '-'  
}
results.append(summary)

# Zapis wyników do pliku CSV
df = pd.DataFrame(results)
df.to_csv('/results/results.csv', index=False)
print(df)

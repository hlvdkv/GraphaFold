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

helix_output_folder = 'results/helix_outputs'
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
            index = int(index_str.strip()) - 1
            code = code_str.strip()
            if '.' in code:
                _, nucleotide_code = code.split('.')
                nucleotide = nucleotide_code[0]
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


if len(sys.argv) < 2:
    print("Usage: python script.py /path/to/folder")
    sys.exit(1)

base_folder = sys.argv[1]
new_amt_folder = os.path.join(base_folder, 'amt')
new_cmt_folder = os.path.join(base_folder, 'cmt')
new_idx_folder = os.path.join(base_folder, 'idx')

has_amt = os.path.isdir(new_amt_folder) and len(os.listdir(new_amt_folder)) > 0
amt_files = sorted(os.listdir(new_amt_folder)) if has_amt else [
    f.replace('.cmt', '.amt') for f in os.listdir(new_cmt_folder) if f.endswith('.cmt')
]

graphs = []
edge_lists = []
labels_list = []
file_names = []

nucleotide_to_onehot = {
    'A': [1, 0, 0, 0],
    'C': [0, 1, 0, 0],
    'G': [0, 0, 1, 0],
    'U': [0, 0, 0, 1],
    None: [0, 0, 0, 0]
}


for amt_file in amt_files:
    cmt_file = amt_file.replace('.amt', '.cmt')
    idx_file = amt_file.replace('.amt', '.idx')
    cmt_file_path = os.path.join(new_cmt_folder, cmt_file)
    idx_file_path = os.path.join(new_idx_folder, idx_file)
    if not os.path.exists(cmt_file_path) or not os.path.exists(idx_file_path):
        print(f'Missing required file(s) for {amt_file}')
        continue

    cmt_matrix = load_matrix(cmt_file_path)

    if has_amt:
        amt_matrix = load_matrix(os.path.join(new_amt_folder, amt_file))
        noncanonical_matrix = np.where(amt_matrix > 1, 1, 0)
        neighborhood_matrix = np.where(cmt_matrix == -1, 1, 0)
    else:
        amt_matrix = np.zeros_like(cmt_matrix)
        noncanonical_matrix = np.zeros_like(cmt_matrix)
        neighborhood_matrix = np.where(cmt_matrix == -1, 1, 0)

    node_nucleotides_dict, max_index_idx = load_idx_file(idx_file_path)

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

    src, dst = np.where(cmt_matrix == 1)
    g = dgl.graph((src, dst), num_nodes=num_nodes)
    num_edges = g.number_of_edges()
    print("edges150", num_edges)
    neigh_src, neigh_dst = np.where(neighborhood_matrix == 1)
    g.add_edges(neigh_src, neigh_dst)
    
    num_edges = g.number_of_edges()
    print("edges155", num_edges)


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
    count=0
    for idx_edge, (i, j) in enumerate(zip(u, v)):
        if noncanonical_matrix[i, j] == 1:
            edge_features[idx_edge] = torch.tensor([1, 1])
        elif cmt_matrix[i, j] == 1:
            edge_features[idx_edge] = torch.tensor([1, 0])
        else:
            edge_features[idx_edge] = torch.tensor([0, 1])
        count+=1

    print("count:", count)
    g.edata['feat'] = edge_features

    # positive_indices = []
    # negative_indices = []
    # for i in range(num_nodes):
    #     for j in range(num_nodes):
    #         if i == j:
    #             continue
    #         if noncanonical_matrix[i, j] == 1:
    #             positive_indices.append((i, j))
    #         else:
    #             nucleotide_i = node_nucleotides_dict.get(i, None)
    #             nucleotide_j = node_nucleotides_dict.get(j, None)
    #             if nucleotide_i is None or nucleotide_j is None:
    #                 continue
    #             negative_indices.append((i, j))

    idxs = []
    for i in range(num_nodes):
        ok=True
        for j in range(num_nodes):
            if cmt_matrix[i, j] > 0:
                ok=False
                break
        if ok:
            idxs.append(i)
    print("idxs:", idxs)
    print("len idxs:", len(idxs))
    positive_indices = []
    negative_indices = []


    for i, vali in enumerate(idxs):
        for j, valj in enumerate(idxs):
            if i == j:
                continue
            if cmt_matrix[vali, valj] == -1:
                # negative_indices.append((i, j))
                continue
            else:
                positive_indices.append((vali, valj))

    print("positive_indices:", len(positive_indices))
    print("negative_indices:", len(negative_indices))
    positive_edges = torch.tensor(positive_indices, dtype=torch.long) if len(positive_indices) > 0 else torch.empty((0, 2), dtype=torch.long)
    negative_edges = torch.tensor(negative_indices, dtype=torch.long) if len(negative_indices) > 0 else torch.empty((0, 2), dtype=torch.long)

    # if len(positive_edges) == 0 or len(negative_edges) == 0:
    #     continue

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
    print("combined_edges:", combined_edges)
    print("len combined_edges:", len(combined_edges))
    graphs.append(g)
    edge_lists.append(combined_edges)
    labels_list.append(edge_labels)
    file_names.append(amt_file)

dataset = GraphDataset(graphs, edge_lists, labels_list, file_names)
test_loader = DataLoader(dataset, batch_size=1, collate_fn=collate)

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
            nn.Linear(hidden_feats, out_feats),
            nn.Softmax(dim=1)
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
        # print("shape edge_inputs:", edge_inputs.shape)
        # print("edge_inputs:", edge_inputs)
        logits = self.edge_predictor(edge_inputs)
        return logits

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GNNModel(in_feats=4, edge_feats=2, hidden_feats=16, out_feats=2)
model.load_state_dict(torch.load('checkpoints/model_v4.pth', map_location=device))
model.to(device)
model.eval()

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

results = []
all_accuracy = []
all_precision = []
all_recall = []
all_f1 = []
all_inf = []

with torch.no_grad():
    for batched_graph, combined_edge_list, combined_labels, file_names_batch in test_loader:
        batched_graph = batched_graph.to(device)
        combined_edge_list = combined_edge_list.to(device)
        combined_labels = combined_labels.to(device)

        logits = model(batched_graph, combined_edge_list)
        raw_logits = logits.cpu().numpy().copy()
        raw_combined = combined_edge_list.cpu().numpy().copy()
        print("length logits:", len(logits))
        print(logits)
        print(len(logits) == len(combined_edge_list))
        # sort logits by column 1
        logits_1 = logits[:, 1]
        combined_copy = combined_edge_list.cpu().numpy().copy()
        
        order = np.arange(len(logits_1))
        results = []
        indexes = []
        while len(combined_copy) > 0:
            max_index = np.argmax(logits_1)
            results.append(combined_copy[max_index])
            selected_pair = combined_copy[max_index]

            for i in range(len(combined_copy)-1, -1, -1):
                if (combined_copy[i][0] == selected_pair[0] or combined_copy[i][1] == selected_pair[1]) \
                    or (combined_copy[i][0] == selected_pair[1] or combined_copy[i][1] == selected_pair[0]):
                    combined_copy = np.delete(combined_copy, i, axis=0)
                    logits_1 = np.delete(logits_1, i)
        
        dict_res = {}
        for x, y in results:
            name = f"{x}_{y}"
            name2 = f"{y}_{x}"
            if name not in dict_res:
                dict_res[name] = 1
            if name2 not in dict_res:
                dict_res[name2] = 1
        
        index=0
        for x, y in combined_edge_list.cpu().numpy():
            name = f"{x}_{y}"
            if name in dict_res:
                logits[index, 0] = 0
                logits[index, 1] = 1
            else:
                logits[index, 0] = 1
                logits[index, 1] = 0
            index+=1
            
            

        # print("results:", np.array(results))
        # predicted = np.zeros(logits.shape)
        # predicted[:, 0] = 1
        

        # print("predicted:", predicted)
        _, predicted = torch.max(logits, 1)
        print('-----')
        # print(predicted)

        graphs_batch = dgl.unbatch(batched_graph)
        num_nodes_list = [g.number_of_nodes() for g in graphs_batch]
        cum_nodes = np.cumsum([0] + num_nodes_list)

        for i, graph in enumerate(graphs_batch):
            start = cum_nodes[i]
            end = cum_nodes[i+1]
            mask = (combined_edge_list[:, 0] >= start) & (combined_edge_list[:, 0] < end)
            edge_list = combined_edge_list[mask] - start
            true_labels = combined_labels[mask]
            predicted_labels = predicted[mask]

            predicted_edges = np.array([tuple(edge.cpu().numpy()) for edge, label in zip(edge_list, predicted_labels) if label == 1])
            print("len true_labels:", len(true_labels))
            print("len predicted_edges:", len(predicted_edges))
            print(f"Predicted edges for {file_names_batch[i]}: {predicted_edges}")

            print("Edges_logits")
            edge_logits = np.array([tuple([tuple(edge), tuple(log)]) for edge, log in zip(raw_combined, raw_logits, strict=True)])
            edge_logits = edge_logits.reshape(-1, 4)
            print(edge_logits)
            # save edge_logits to csv file
            
            np.savetxt("logits.csv", edge_logits, delimiter=",", fmt="%s")
            print(edge_logits.shape)
            print("-----")
            if has_amt:
                # gt_edges = [tuple(edge.cpu().numpy()) for edge, label in zip(edge_list, true_labels) if label == 1]
                # gt_edges_sorted = set(map(lambda x: tuple(sorted(x)), gt_edges))
                predicted_edges = set([(int(x),int(y)) for x,y in predicted_edges if x < y])
                pred_edges_sorted = set(map(lambda x: tuple(sorted(x)), predicted_edges))
                
                amt_matrix = load_matrix(os.path.join(new_amt_folder, amt_file))
                nc_x, nc_y = np.where(amt_matrix > 1)
                gt_edges = set([(int(x),int(y)) for x,y in zip(nc_x, nc_y) if x < y])
                gt_edges_sorted = set(map(lambda x: tuple(sorted(x)), gt_edges))


                # print("gt_edges:", gt_edges_sorted)
                # print("pred_edges_sorted:", pred_edges_sorted)

                tp = len(gt_edges_sorted & pred_edges_sorted)
                fn = len(gt_edges_sorted - pred_edges_sorted)
                fp = len(pred_edges_sorted - gt_edges_sorted)

                ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
                tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
                inf_metric = np.sqrt(ppv * tpr) if (ppv * tpr) > 0 else 0

                all_accuracy.append(np.mean(predicted_labels.cpu().numpy() == true_labels.cpu().numpy()))
                all_precision.append(ppv)
                all_recall.append(tpr)
                all_f1.append(f1_score(true_labels.cpu().numpy(), predicted_labels.cpu().numpy(), zero_division=0))
                all_inf.append(inf_metric)

                results.append({
                    'File name': file_names_batch[i],
                    'TP': tp,
                    'FN': fn,
                    'FP': fp,
                    'PPV': ppv,
                    'TPR': tpr,
                    'INF': inf_metric
                })

                helix_data = []
                for edge in gt_edges_sorted:
                    if edge in pred_edges_sorted:
                        helix_data.append(f"{edge[0]},{edge[1]},PredictedGoodNonCanonical,1.0")
                for edge in pred_edges_sorted - gt_edges_sorted:
                    helix_data.append(f"{edge[0]},{edge[1]},PredictedBadNonCanonical,1.0")
                for edge in gt_edges_sorted - pred_edges_sorted:
                    helix_data.append(f"{edge[0]},{edge[1]},NotPredictedNonCanonical,1.0")

                helix_file_path = os.path.join(helix_output_folder, f"{file_names_batch[i]}.helix")
                with open(helix_file_path, 'w') as f:
                    f.write("\n".join(helix_data) + "\n")

if has_amt:
    mean_accuracy = np.mean(all_accuracy)
    mean_precision = np.mean(all_precision)
    mean_recall = np.mean(all_recall)
    mean_f1 = np.mean(all_f1)
    mean_inf = np.mean(all_inf)

    print("Mean accuracy: {:.4f}".format(mean_accuracy))
    print("Mean precision: {:.4f}".format(mean_precision))
    print("Mean recall: {:.4f}".format(mean_recall))
    print("Mean f1: {:.4f}".format(mean_f1))
    print("Mean inf: {:.4f}".format(mean_inf))

    summary = {
        'File name': 'Mean',
        'TP': '-',
        'FN': '-',
        'FP': '-',
        'PPV': mean_precision,
        'TPR': mean_recall,
        'INF': mean_inf
    }
    results.append(summary)

    df = pd.DataFrame(results)
    df.to_csv('results/results.csv', index=False)
#!/usr/bin/env python3

import os, math, csv, argparse
import numpy as np, torch, dgl
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
from utils import parse_dot2out

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model
from dgl.nn import NNConv
class GNN(nn.Module):
    def __init__(self, h: int = 128, edge_dim: int = 3):
        super().__init__()
        self.lin, self.bn0 = nn.Linear(4, h), nn.BatchNorm1d(h)
        edge_net = nn.Sequential(nn.Linear(edge_dim, h * h),
                                 nn.ReLU(), nn.BatchNorm1d(h * h))
        self.g1, self.g2 = NNConv(h, h, edge_net, "mean"), NNConv(h, h, edge_net, "mean")
        self.bn1, self.bn2 = nn.BatchNorm1d(h), nn.BatchNorm1d(h)
        self.pred = nn.Sequential(nn.Linear(2 * h, h), nn.ReLU(),
                                  nn.Dropout(0.25), nn.Linear(h, 1))

    def forward(self, g: dgl.DGLGraph, idx: torch.Tensor):
        h0 = F.relu(self.bn0(self.lin(g.ndata["feat"])))
        h1 = F.relu(self.bn1(self.g1(g, h0, g.edata["feat"])))
        h1 = h0 + F.dropout(h1, 0.15, self.training)
        h2 = F.relu(self.bn2(self.g2(g, h1, g.edata["feat"])))
        h2 = h1 + F.dropout(h2, 0.15, self.training)
        s, d = idx[:, 0], idx[:, 1]
        return self.pred(torch.cat([h2[s], h2[d]], 1)).squeeze(-1)

# helpers 
def load_matrix(p): return np.loadtxt(p, delimiter=",")

def load_idx(path):
    nuc, idxs = {}, []
    for ln in open(path):
        if not ln.strip(): continue
        i, code = ln.strip().split(",")
        i = int(i) - 1
        idxs.append(i)
        nuc[i] = code.split(".")[1][0] if "." in code else None
    return nuc, max(idxs) if idxs else -1

def build_graph(base, dirs):
    cmt = load_matrix(os.path.join(dirs["cmt"], f"{base}.cmt"))
    amt = load_matrix(os.path.join(dirs["amt"], f"{base}.amt"))
    noncan = (amt > 1).astype(int)
    neigh  = (amt == -1).astype(int)
    nuc_map, idx_max = load_idx(os.path.join(dirs["idx"], f"{base}.idx"))
    N = max(idx_max + 1, cmt.shape[0])

    def pad(m, size):
        if m.shape[0] >= size: return m
        out = np.zeros((size, size), dtype=m.dtype)
        out[:m.shape[0], :m.shape[1]] = m
        return out
    cmt, noncan, neigh = (pad(m, N) for m in (cmt, noncan, neigh))

    src, dst, efeat = [], [], []
    def add(mat, vec):
        i, j = np.where(mat == 1)
        src.extend(i); dst.extend(j); efeat.extend([vec]*len(i))
        src.extend(j); dst.extend(i); efeat.extend([vec]*len(i))
    add(noncan, [1,0,0]); add(cmt,[0,1,0]); add(neigh,[0,0,1])

    g = dgl.graph((src, dst), num_nodes=N)
    g.edata["feat"] = torch.tensor(efeat, dtype=torch.float32)
    onehot = {"A":[1,0,0,0],"C":[0,1,0,0],"G":[0,0,1,0],"U":[0,0,0,1],None:[0,0,0,0]}
    g.ndata["feat"] = torch.tensor([onehot[nuc_map.get(i)] for i in range(N)],
                                   dtype=torch.float32)
    return g, noncan

def batched_predict(g, model, batch=50_000):
    N = g.num_nodes()
    pairs = torch.combinations(torch.arange(N), r=2)
    probs = []
    for i in range(0, len(pairs), batch):
        idx = pairs[i:i+batch].to(device)
        with torch.no_grad(), torch.cuda.amp.autocast():
            logits = model(g.to(device), idx)
        probs.append(torch.sigmoid(logits).cpu())
    return pairs.cpu().numpy(), torch.cat(probs).numpy()

# main 
def main(args):
    if os.path.exists(args.input):
        print(f"Parsing input file {args.input} ...")
        idx_file, cmt_file = parse_dot2out(args.input)
        idx_dir = os.path.dirname(idx_file)
        cmt_dir = os.path.dirname(cmt_file)
    ckpt  = torch.load(args.model, map_location="cpu")
    model = GNN().to(device); model.load_state_dict(ckpt["state"]); model.eval()
    thr   = ckpt["threshold"]

    dirs = {"amt": args.amt_dir, "cmt": cmt_dir, "idx": idx_dir}
    os.makedirs(args.out_dir, exist_ok=True)

    all_metrics = []

    with open(args.csv_out, "w", newline="") as csv_f:
        writer = csv.writer(csv_f)
        writer.writerow(["graph", "accuracy", "precision", "recall",
                         "F1", "inf", "pos", "neg"])

        for fname in tqdm(os.listdir(args.amt_dir), desc="graphs"):
            if not fname.endswith(".amt"): continue
            base = fname[:-4]
            if not all(os.path.exists(os.path.join(v, f"{base}.{ext}"))
                       for v, ext in ((cmt_dir,"cmt"), (idx_dir,"idx"))):
                print(f"[skip] {base} no files in the directory"); continue

            g, noncan = build_graph(base, dirs)
            pairs, probs = batched_predict(g, model, args.batch)
            preds  = (probs >= thr)

            # csv
            out = np.column_stack([pairs, probs, preds.astype(int)])
            np.savetxt(os.path.join(args.out_dir, f"{base}_pred.csv"),
                       out, fmt=["%d","%d","%.6f","%d"],
                       delimiter=",", header="i,j,prob,label", comments="")
            # TODO: output to dot file or bpseq

# CLI 
if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--model", default="model.pth")
    a.add_argument("--input", default="example.dot", help="Input file in dot format")
    a.add_argument("--out_dir", default="predictions_full")
    a.add_argument("--batch",   type=int, default=1)
    main(a.parse_args())

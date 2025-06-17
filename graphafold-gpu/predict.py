#!/usr/bin/env python3

import os, math, csv, argparse
import numpy as np, torch, dgl
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm

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
    ckpt  = torch.load(args.model, map_location="cpu")
    model = GNN().to(device); model.load_state_dict(ckpt["state"]); model.eval()
    thr   = ckpt["threshold"]

    dirs = {"amt": args.amt_dir, "cmt": args.cmt_dir, "idx": args.idx_dir}
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
                       for v, ext in ((args.cmt_dir,"cmt"), (args.idx_dir,"idx"))):
                print(f"[skip] {base} – brak plików"); continue

            g, noncan = build_graph(base, dirs)
            pairs, probs = batched_predict(g, model, args.batch)

            preds  = (probs >= thr)
            labels = np.array([noncan[i, j] for i, j in pairs])

            acc = accuracy_score(labels, preds)
            P   = precision_score(labels, preds, zero_division=0)
            R   = recall_score(labels, preds, zero_division=0)
            F1  = f1_score(labels, preds, zero_division=0)
            inf = math.sqrt(P * R) if P * R > 0 else 0.0

            pos_cnt = labels.sum(); neg_cnt = len(labels) - pos_cnt
            print(f"\n {base}")
            print(f"  Accuracy={acc:.3f}  Precision={P:.3f}  Recall={R:.3f}  "
                  f"F1={F1:.3f}  inf={inf:.3f}  Pos={int(pos_cnt):>3}, Neg={int(neg_cnt):>4}")

            writer.writerow([base, f"{acc:.3f}", f"{P:.3f}", f"{R:.3f}",
                             f"{F1:.3f}", f"{inf:.3f}", int(pos_cnt), int(neg_cnt)])

            all_metrics.append((acc, P, R, F1, inf))

            # csv
            out = np.column_stack([pairs, probs, preds.astype(int)])
            np.savetxt(os.path.join(args.out_dir, f"{base}_pred.csv"),
                       out, fmt=["%d","%d","%.6f","%d"],
                       delimiter=",", header="i,j,prob,label", comments="")

        # średnie 
        accs, Ps, Rs, F1s, infs = map(np.mean, zip(*all_metrics))
        print("\nŚrednie metryki:")
        print(f"  Accuracy:  {accs:.3f}")
        print(f"  Precision: {Ps:.3f}")
        print(f"  Recall:    {Rs:.3f}")
        print(f"  F1-score:  {F1s:.3f}")
        print(f"  inf-score: {infs:.3f}")

        writer.writerow(["GLOBAL", f"{accs:.3f}", f"{Ps:.3f}", f"{Rs:.3f}",
                         f"{F1s:.3f}", f"{infs:.3f}", "-", "-"])

# CLI 
if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--model", default="model.pth")
    a.add_argument("--amt_dir", default="TestSet/amt")
    a.add_argument("--cmt_dir", default="TestSet/cmt")
    a.add_argument("--idx_dir", default="TestSet/idx")
    a.add_argument("--out_dir", default="predictions_full")
    a.add_argument("--batch",   type=int, default=50_000)
    a.add_argument("--csv_out", default="metrics_full.csv",
                   help="plik z metrykami per-graf + global")
    main(a.parse_args())

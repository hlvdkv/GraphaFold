#!/usr/bin/env python3

import os, math, csv, argparse
import numpy as np, torch, dgl
import torch.nn.functional as F
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model
from dgl.nn import NNConv
class GNN(torch.nn.Module):
    def __init__(self, h: int = 128, edge_dim: int = 3):
        super().__init__()
        self.lin, self.bn0 = torch.nn.Linear(4, h), torch.nn.BatchNorm1d(h)
        edge_net = torch.nn.Sequential(
            torch.nn.Linear(edge_dim, h * h), torch.nn.ReLU(),
            torch.nn.BatchNorm1d(h * h))
        self.g1, self.g2 = NNConv(h, h, edge_net, "mean"), NNConv(h, h, edge_net, "mean")
        self.bn1, self.bn2 = torch.nn.BatchNorm1d(h), torch.nn.BatchNorm1d(h)
        self.pred = torch.nn.Sequential(
            torch.nn.Linear(2 * h, h), torch.nn.ReLU(),
            torch.nn.Dropout(0.25), torch.nn.Linear(h, 1))

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

def process_file(base: str, dirs: dict[str, str], win: int = 15):
    """Buduje graf + listę krawędzi w oknach."""
    cmt   = load_matrix(os.path.join(dirs["cmt"], f"{base}.cmt"))
    amt   = load_matrix(os.path.join(dirs["amt"], f"{base}.amt"))
    noncan = (amt > 1).astype(int)
    neigh  = (amt == -1).astype(int)
    nuc_map, idx_max = load_idx(os.path.join(dirs["idx"], f"{base}.idx"))
    N = max(idx_max + 1, cmt.shape[0])

    def pad(m):
        if m.shape[0] >= N: return m
        out = np.zeros((N, N), dtype=m.dtype)
        out[:m.shape[0], :m.shape[1]] = m
        return out
    cmt, noncan, neigh = (pad(m) for m in (cmt, noncan, neigh))

    allowed = np.zeros((N, N), bool)
    xs, ys = np.where(np.triu(cmt, 1))
    for x, y in zip(xs, ys):
        xl, xh = max(0, x - win), min(N, x + win)
        yl, yh = max(0, y - win), min(N, y + win)
        allowed[xl:xh, xl:xh] = True
        allowed[yl:yh, yl:yh] = True
        allowed[xl:xh, yl:yh] = True
        allowed[yl:yh, xl:xh] = True
    allowed = np.triu(allowed, 1)

    pos, neg = [], []
    for i in range(N):
        for j in range(i + 1, N):
            if not allowed[i, j]: continue
            if noncan[i, j]:
                pos.append((i, j))
            elif not (cmt[i, j] or neigh[i, j]):
                neg.append((i, j))

    total_pos  = int(np.triu(noncan, 1).sum())
    hidden_pos = total_pos - len(pos)

    src, dst, efeat = [], [], []
    def add(mat, vec):
        ii, jj = np.where(mat == 1)
        src.extend(ii); dst.extend(jj); efeat.extend([vec]*len(ii))
        src.extend(jj); dst.extend(ii); efeat.extend([vec]*len(ii))
    add(noncan, [1,0,0]); add(cmt,[0,1,0]); add(neigh,[0,0,1])

    g = dgl.graph((src, dst), num_nodes=N)
    g.edata["feat"] = torch.tensor(efeat, dtype=torch.float32)
    onehot = {"A":[1,0,0,0],"C":[0,1,0,0],"G":[0,0,1,0],"U":[0,0,0,1],None:[0,0,0,0]}
    g.ndata["feat"] = torch.tensor([onehot[nuc_map.get(i)] for i in range(N)], dtype=torch.float32)

    edges  = torch.tensor(pos + neg, dtype=torch.long)
    labels = torch.tensor([1]*len(pos) + [0]*len(neg), dtype=torch.float32)
    return g, edges, labels, hidden_pos

# main 
def predict_all_files(args):
    # CSV output
    with open(args.csv_out, "w", newline="") as csv_f:
        writer = csv.writer(csv_f)
        writer.writerow(["graph", "precision", "recall", "F1", "inf",
                         "pos_in_window", "hidden_pos"])

        ckpt  = torch.load(args.model, map_location="cpu")
        model = GNN().to(device); model.load_state_dict(ckpt["state"]); model.eval()
        thr   = ckpt["threshold"]

        dirs  = {"amt": args.amt_dir, "cmt": args.cmt_dir, "idx": args.idx_dir}
        agg   = dict(tp=0, fp=0, fn_in=0, fn_hidden=0)

        for base in tqdm(sorted(f[:-4] for f in os.listdir(args.amt_dir)
                                if f.endswith(".amt")), desc="graphs"):
            try:
                g, edges, labels_cpu, hpos = process_file(base, dirs, win=args.window)
            except Exception as e:
                print("Skipped", base, "→", e); continue

            g, edges = g.to(device), edges.to(device)
            labels   = labels_cpu.to(device)

            with torch.no_grad(), torch.cuda.amp.autocast():
                preds = (torch.sigmoid(model(g, edges)) >= thr).long()

            tp   = int(((preds==1) & (labels==1)).sum())
            fp   = int(((preds==1) & (labels==0)).sum())
            fn_in = int(((preds==0) & (labels==1)).sum())
            fn_tot = fn_in + hpos

            P = tp / (tp + fp) if tp + fp else 0.0
            R = tp / (tp + fn_tot) if tp + fn_tot else 0.0
            F1 = (2 * P * R) / (P + R) if P + R else 0.0
            INF = math.sqrt(P * R) if P * R else 0.0

            pos_in_window = int(labels_cpu.sum())

            # log – CLI
            print(f"\n {base}")
            print(f"  P={P:.3f}  R={R:.3f}  F1={F1:.3f}  inf={INF:.3f}  "
                  f"pos_in_window={pos_in_window}  hidden_pos={hpos}")

            # log – CSV
            writer.writerow([base, f"{P:.3f}", f"{R:.3f}",
                             f"{F1:.3f}", f"{INF:.3f}",
                             pos_in_window, hpos])

            agg["tp"] += tp; agg["fp"] += fp
            agg["fn_in"] += fn_in; agg["fn_hidden"] += hpos

        # global summary 
        tp, fp = agg["tp"], agg["fp"]
        fn     = agg["fn_in"] + agg["fn_hidden"]
        P = tp / (tp + fp) if tp + fp else 0.0
        R = tp / (tp + fn) if tp + fn else 0.0
        F1= (2 * P * R) / (P + R) if P + R else 0.0
        INF = math.sqrt(P * R) if P * R else 0.0

        print("\n================ GLOBAL =================")
        print(f"  TP={tp}  FP={fp}  FN={fn}")
        print(f"  Precision={P:.3f}  Recall={R:.3f}  F1={F1:.3f}  inf={INF:.3f}")

        writer.writerow(["GLOBAL", f"{P:.3f}", f"{R:.3f}",
                         f"{F1:.3f}", f"{INF:.3f}", "-", "-"])

# CLI
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",   default="simple-model.pth")
    ap.add_argument("--amt_dir", default="FullGraphs/amt")
    ap.add_argument("--cmt_dir", default="FullGraphs/cmt")
    ap.add_argument("--idx_dir", default="FullGraphs/idx")
    ap.add_argument("--window",  type=int, default=15, help="pół-szerokość okna (nt)")
    ap.add_argument("--csv_out", default="metrics_windows.csv",
                    help="ścieżka pliku CSV z metrykami")
    predict_all_files(ap.parse_args())

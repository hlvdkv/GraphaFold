

import os, math, csv, argparse, numpy as np, torch, dgl
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



from rinalmo.pretrained import get_pretrained_model
from dgl.nn import NNConv


class GNN(nn.Module):
    def __init__(self, h: int = 128, edge_dim: int = 3):
        super().__init__()

        self.rinalmo, self.alphabet = get_pretrained_model("giga-v1")
        self.rinalmo.eval()
        for p in self.rinalmo.parameters():
            p.requires_grad = False

        self.seq_proj = nn.Sequential(
            nn.Linear(1280, h),
            nn.ReLU(),
            nn.Linear(h, h),
            nn.ReLU(),
        )
        self.bn0 = nn.BatchNorm1d(h)

        edge_net = nn.Sequential(
            nn.Linear(edge_dim, h * h),
            nn.ReLU(),
            nn.BatchNorm1d(h * h),
        )
        self.g1, self.g2 = NNConv(h, h, edge_net, "mean"), NNConv(h, h, edge_net, "mean")
        self.bn1, self.bn2 = nn.BatchNorm1d(h), nn.BatchNorm1d(h)

        self.pred = nn.Sequential(
            nn.Linear(2 * h, h),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(h, 1),
        )
        self.dropout = nn.Dropout(0.15)

    @torch.no_grad()
    def _encode(self, sequences: list[str]) -> torch.Tensor:
        
        tokens = torch.tensor(
            self.alphabet.batch_tokenize(sequences), dtype=torch.int64, device=device
        )
        with torch.cuda.amp.autocast():
            rep = self.rinalmo(tokens)["representation"]     

        reps = []
        for i, seq in enumerate(sequences):
            L = len(seq)
            reps.append(rep[i, 1 : 1 + L])        
        rep_cat = torch.cat(reps, dim=0)          
        return self.seq_proj(rep_cat)             

    def forward(self, g: dgl.DGLGraph, idx_pairs: torch.Tensor, sequences: list[str]):
        node_feats = self._encode(sequences)      
        h0 = F.relu(self.bn0(node_feats))
        h1 = F.relu(self.bn1(self.g1(g, h0, g.edata["feat"])))
        h1 = h0 + self.dropout(h1)
        h2 = F.relu(self.bn2(self.g2(g, h1, g.edata["feat"])))
        h2 = h1 + self.dropout(h2)

        s, d = idx_pairs[:, 0], idx_pairs[:, 1]
        return self.pred(torch.cat([h2[s], h2[d]], 1)).squeeze(-1)


def load_matrix(p): return np.loadtxt(p, delimiter=",")


def load_idx(path):
    nuc, idxs = {}, []
    for ln in open(path):
        if not ln.strip():
            continue
        i, code = ln.strip().split(",")
        i = int(i) - 1
        idxs.append(i)
        nuc[i] = code.split(".")[1][0] if "." in code else None
    return nuc, max(idxs) if idxs else -1


def build_graph_and_seq(base, dirs):
    
    cmt = load_matrix(os.path.join(dirs["cmt"], f"{base}.cmt"))
    amt = load_matrix(os.path.join(dirs["amt"], f"{base}.amt"))
    noncan = (amt > 1).astype(int)
    neigh  = (amt == -1).astype(int)
    nuc_map, idx_max = load_idx(os.path.join(dirs["idx"], f"{base}.idx"))
    N = max(idx_max + 1, cmt.shape[0])

    def pad(m, size):
        if m.shape[0] >= size:
            return m
        out = np.zeros((size, size), dtype=m.dtype)
        out[: m.shape[0], : m.shape[1]] = m
        return out
    cmt, noncan, neigh = (pad(m, N) for m in (cmt, noncan, neigh))

    src, dst, efeat = [], [], []

    def add(mat, vec):
        i, j = np.where(mat == 1)
        src.extend(i.tolist()); dst.extend(j.tolist()); efeat.extend([vec] * len(i))
        src.extend(j.tolist()); dst.extend(i.tolist()); efeat.extend([vec] * len(i))

    add(noncan, [1, 0, 0])
    add(cmt,    [0, 1, 0])
    add(neigh,  [0, 0, 1])

    g = dgl.graph((src, dst), num_nodes=N)
    g.edata["feat"] = torch.tensor(efeat, dtype=torch.float32)
    # one-hot nieużywane – zostawiamy jako placeholder
    onehot = {"A":[1,0,0,0], "C":[0,1,0,0], "G":[0,0,1,0], "U":[0,0,0,1], None:[0,0,0,0]}
    g.ndata["feat"] = torch.tensor([onehot[nuc_map.get(i)] for i in range(N)],
                                   dtype=torch.float32)

    seq = "".join(nuc_map.get(i, "N") for i in range(N))
    return g, seq, noncan


def batched_predict(g: dgl.DGLGraph, seq: str, model: nn.Module,
                    batch_size: int = 50_000):
    N = g.num_nodes()
    pairs = torch.combinations(torch.arange(N), r=2)
    probs = []

    g = g.to(device)
    for i in range(0, len(pairs), batch_size):
        idx = pairs[i : i + batch_size].to(device)
        with torch.no_grad(), torch.cuda.amp.autocast():
            logits = model(g, idx, [seq])
        probs.append(torch.sigmoid(logits).cpu())

    return pairs.cpu().numpy(), torch.cat(probs).numpy()


def main(args):
    ckpt = torch.load(args.model, map_location="cpu")
    model = GNN().to(device)
    model.load_state_dict(ckpt["state"])
    model.eval()
    thr = ckpt["threshold"]

    dirs = {"amt": args.amt_dir, "cmt": args.cmt_dir, "idx": args.idx_dir}
    os.makedirs(args.out_dir, exist_ok=True)
    helix_output_folder = os.path.join( "helix")
    os.makedirs(helix_output_folder, exist_ok=True)

    all_metrics = []

    with open(args.csv_out, "w", newline="") as csv_f:
        writer = csv.writer(csv_f)
        writer.writerow(["graph", "accuracy", "precision", "recall",
                         "F1", "inf", "pos", "neg"])

        for fname in tqdm(os.listdir(args.amt_dir), desc="graphs"):
            if not fname.endswith(".amt"):
                continue
            base = fname[:-4]
            if not all(
                os.path.exists(os.path.join(v, f"{base}.{ext}"))
                for v, ext in ((args.cmt_dir, "cmt"), (args.idx_dir, "idx"))
            ):
                print(f"[skip] {base} – brak plików")
                continue

            g, seq, noncan = build_graph_and_seq(base, dirs)
            pairs, probs = batched_predict(g, seq, model, args.batch)

            preds = (probs >= thr)
            labels = np.array([noncan[i, j] for i, j in pairs])

            acc = accuracy_score(labels, preds)
            P   = precision_score(labels, preds, zero_division=0)
            R   = recall_score(labels, preds, zero_division=0)
            F1  = f1_score(labels, preds, zero_division=0)
            inf = math.sqrt(P * R) if P * R > 0 else 0.0

            pos_cnt = labels.sum()
            neg_cnt = len(labels) - pos_cnt
            print(f"\n {base}")
            print(f"  Accuracy={acc:.3f}  Precision={P:.3f}  Recall={R:.3f}  "
                  f"F1={F1:.3f}  inf={inf:.3f}  Pos={int(pos_cnt):>3}, Neg={int(neg_cnt):>4}")

            writer.writerow([base, f"{acc:.3f}", f"{P:.3f}", f"{R:.3f}",
                             f"{F1:.3f}", f"{inf:.3f}", int(pos_cnt), int(neg_cnt)])

            all_metrics.append((acc, P, R, F1, inf))

            out = np.column_stack([pairs, probs, preds.astype(int)])
            np.savetxt(os.path.join(args.out_dir, f"{base}_pred.csv"),
                       out, fmt=["%d","%d","%.6f","%d"],
                       delimiter=",", header="i,j,prob,label", comments="")

            edges_np = pairs
            labels_np = labels
            preds_np = preds

            gt_edges = [tuple(edge) for edge, lab in zip(edges_np, labels_np) if lab == 1]
            pred_edges = [tuple(edge) for edge, pred in zip(edges_np, preds_np) if pred == 1]

            def sort_edge(edge):
                return tuple(sorted(edge))

            gt_sorted = set(map(sort_edge, gt_edges))
            pred_sorted = set(map(sort_edge, pred_edges))

            good = gt_sorted & pred_sorted
            bad = pred_sorted - gt_sorted
            missed = gt_sorted - pred_sorted

            helix_lines = []
            for edge in good:
                helix_lines.append(f"{edge[0]},{edge[1]},PredictedGoodNonCanonical,1.0")
            for edge in bad:
                helix_lines.append(f"{edge[0]},{edge[1]},PredictedBadNonCanonical,1.0")
            for edge in missed:
                helix_lines.append(f"{edge[0]},{edge[1]},NotPredictedNonCanonical,1.0")

            helix_file_path = os.path.join(helix_output_folder, f"{base}.helix")
            with open(helix_file_path, 'w') as hf:
                hf.write("\n".join(helix_lines) + "\n")

        # średnie globalne 
        accs, Ps, Rs, F1s, infs = map(np.mean, zip(*all_metrics))
        print("\nŚrednie metryki:")
        print(f"  Accuracy:  {accs:.3f}")
        print(f"  Precision: {Ps:.3f}")
        print(f"  Recall:    {Rs:.3f}")
        print(f"  F1-score:  {F1s:.3f}")
        print(f"  inf-score: {infs:.3f}")

        writer.writerow(["GLOBAL", f"{accs:.3f}", f"{Ps:.3f}", f"{Rs:.3f}",
                         f"{F1s:.3f}", f"{infs:.3f}", "-", "-"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="rinalmo-model.pth",
                    help="checkpoint wytrenowanego modelu RiNALMo-GNN (.pth)")
    ap.add_argument("--amt_dir", default="CaspRNA/amt")
    ap.add_argument("--cmt_dir", default="CaspRNA/cmt")
    ap.add_argument("--idx_dir", default="CaspRNA/idx")
    ap.add_argument("--out_dir", default="predictions_full_rinalmo")
    ap.add_argument("--batch",   type=int, default=50_000,
                    help="ile par (i,j) przewidujemy jednorazowo na GPU")
    ap.add_argument("--csv_out", default="metrics_full_rinalmo.csv")
    args = ap.parse_args()
    main(args)

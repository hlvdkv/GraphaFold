#!/usr/bin/env python3

import os, math, csv, argparse
import numpy as np, torch, dgl
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
from utils import parse_dot2out, fill_mat_with_pairs, mat_to_bpseq
from rinalmo.pretrained import get_pretrained_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model
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

def build_graph(base, dirs, neigh):
    cmt = load_matrix(os.path.join(dirs["cmt"], f"{base}.cmt"))
    cmt = np.array(cmt, dtype=int)
    print(cmt.shape)
    nuc_map, idx_max = load_idx(os.path.join(dirs["idx"], f"{base}.idx"))
    N = max(idx_max + 1, cmt.shape[0])

    def pad(m, size):
        print(m)
        if m.shape[0] >= size: return m
        out = np.zeros((size, size), dtype=m.dtype)
        out[:m.shape[0], :m.shape[1]] = m
        return out


    src, dst, efeat = [], [], []
    def add(mat, vec, neigh=False):
        if not neigh:
            i, j = np.where(mat == 1)
        else:
            i, j = mat
        src.extend(i); dst.extend(j); efeat.extend([vec]*len(i))
        src.extend(j); dst.extend(i); efeat.extend([vec]*len(i))
    add(cmt,[0,1,0])
    add(neigh,[0,0,1], neigh=True)

    g = dgl.graph((src, dst), num_nodes=N)
    g.edata["feat"] = torch.tensor(efeat, dtype=torch.float32)
    onehot = {"A":[1,0,0,0],"C":[0,1,0,0],"G":[0,0,1,0],"U":[0,0,0,1],None:[0,0,0,0]}
    g.ndata["feat"] = torch.tensor([onehot[nuc_map.get(i)] for i in range(N)],
                                   dtype=torch.float32)
    return g, cmt

def batched_predict(g, model, sequences, batch=50_000):
    N = g.num_nodes()
    pairs = torch.combinations(torch.arange(N), r=2)
    probs = []
    for i in range(0, len(pairs), batch):
        idx = pairs[i:i+batch].to(device)
        with torch.no_grad(), torch.cuda.amp.autocast():
            logits = model(g.to(device), idx, sequences)
        probs.append(torch.sigmoid(logits).cpu())
    return pairs.cpu().numpy(), torch.cat(probs).numpy()

# main 
def main(args):
    if os.path.exists(args.input):
        print(f"Parsing input file {args.input} ...")
        idx_file, cmt_file, neighs, sequences = parse_dot2out(args.input)
        idx_dir = os.path.dirname(idx_file)
        cmt_dir = os.path.dirname(cmt_file)
    ckpt  = torch.load(args.model, map_location="cpu")
    model = GNN().to(device); model.load_state_dict(ckpt["state"]); model.eval()
    thr   = ckpt["threshold"]

    dirs = {"cmt": cmt_dir, "idx": idx_dir}
    os.makedirs(args.out_dir, exist_ok=True)

    all_metrics = []

    
    for fname in tqdm(os.listdir(cmt_dir), desc="graphs"):
        base = fname[:-4]
        if not fname.endswith(".cmt"): continue
        if not all(os.path.exists(os.path.join(v, f"{base}.{ext}"))
                    for v, ext in ((cmt_dir,"cmt"), (idx_dir,"idx"))):
            print(f"[skip] {base} no files in the directory"); continue

        g, cmt = build_graph(base, dirs, neighs)
        pairs, probs = batched_predict(g, model, sequences, args.batch)
        preds  = (probs >= thr)

        # csv
        out = np.column_stack([pairs, probs, preds.astype(int)])
        np.savetxt(os.path.join(args.out_dir, f"{base}_pred.csv"),
                    out, fmt=["%d","%d","%.6f","%d"],
                    delimiter=",", header="i,j,prob,label", comments="")
        
        mat_all = fill_mat_with_pairs(cmt.copy(), pairs[preds])
        bpseq_all = mat_to_bpseq(mat_all, sequences)
        with open(os.path.join(args.out_dir, f"{base}_all.bpseq"), "w") as f:
            f.write(bpseq_all + "\n")
        mat_noncan = fill_mat_with_pairs(np.zeros_like(cmt), pairs[preds])
        bpseq_noncan = mat_to_bpseq(mat_noncan, sequences)
        with open(os.path.join(args.out_dir, f"{base}_noncan.bpseq"), "w") as f:
            f.write(bpseq_noncan + "\n")

# CLI 
if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--model", default="model.pth")
    a.add_argument("--input", default="example.dot", help="Input file in dot format")
    a.add_argument("--out_dir", default="predictions_full")
    a.add_argument("--batch",   type=int, default=1)
    main(a.parse_args())

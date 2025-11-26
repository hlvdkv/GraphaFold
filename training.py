#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, math, random, argparse
import numpy as np
import torch, dgl
from torch import nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, auc, precision_score, recall_score, f1_score
from dgl.nn import NNConv, GATConv
from rinalmo.pretrained import get_pretrained_model


def build_argparser():
    p = argparse.ArgumentParser(description="RNA noncanonical interaction predictor (bi-encoder + reranker)")
    p.add_argument("--data_root", default="TrainingSet")
    p.add_argument("--window", type=int, default=200)
    p.add_argument("--neg_per_pos", type=int, default=25)
    p.add_argument("--neg_local_frac", type=float, default=0.85)
    # model
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--depth", type=int, default=5)
    p.add_argument("--bins", default="15,30,50,100,200")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=70)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--adam_eps", type=float, default=1e-8)
    p.add_argument("--clip_grad", type=float, default=1.0)
    p.add_argument("--label_smoothing", type=float, default=0.05)
    p.add_argument("--cb_posw", action="store_true", help="Użyj class-balanced pos_weight ")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dropout", type=float, default=0.15)
    p.add_argument("--dropout_pred", type=float, default=0.25)
    p.add_argument("--use_layernorm", action="store_true", help="Zamień BatchNorm1d na LayerNorm")
    p.add_argument("--save", default="bimodel.pth")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--no_mixed", action="store_true")
    p.add_argument("--use_log_dist", action="store_true")
    # Transformer
    # p.add_argument("--use_ssm", action="store_true")
    # p.add_argument("--ssm_layers", type=int, default=2)
    # p.add_argument("--ssm_heads", type=int, default=8)
    # p.add_argument("--ssm_dropout", type=float, default=0.1)
    # p.add_argument("--ssm_no_posenc", action="store_true", help="Wyłącz sinusoidalne pozycje w SSM")
    # Etap 1 (bi-encoder)
    p.add_argument("--stage1_epochs", type=int, default=3, help="Liczba epok pre-treningu bi-encodera (0=wyłącz)")
    p.add_argument("--stage1_lr", type=float, default=5e-4)
    p.add_argument("--stage1_neg_per_pos", type=int, default=30)
    p.add_argument("--stage1_topk", type=int, default=5000, help="Top-k kandydatów do rerankingu w ewaluacji (0=wyłącz)")
    p.add_argument("--save_stage1", default="bimodel-stage1.pth")
    p.add_argument("--load_stage1", default="", help="Ścieżka do istniejącego Stage-1; jeśli podana, pomija trening")
    # typ GNN i parametry GAT
    p.add_argument("--gnn_type", choices=["nnconv", "gat", "hybrid"], default="nnconv",
                   help="Encoder grafowy: NNConv (z cechami krawędzi), GAT (uwaga), lub hybrid (naprzemiennie).")
    p.add_argument("--gat_heads", type=int, default=4, help="Liczba głów w GATConv.")
    p.add_argument("--gat_feat_drop", type=float, default=0.0, help="Dropout na cechach w GAT.")
    p.add_argument("--gat_attn_drop", type=float, default=0.0, help="Dropout na wagach uwagi w GAT.")
    p.add_argument("--gat_residual", action="store_true", help="Włącz residual w GATConv.")

    p.add_argument("--rinalmo_max_len", type=int, default=1024,
               help="Maks. długość chunku dla RiNALMo (bez specjalnych tokenów).")
    p.add_argument("--rinalmo_overlap", type=int, default=64,
               help="Zakładka (liczba nt) między chunkami RiNALMo.")

    return p


# Utils
def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def load_matrix(p): return np.loadtxt(p, delimiter=",")

def load_idx(p):
    nuc, idxs = {}, []
    with open(p) as fh:
        for ln in fh:
            if not ln.strip(): continue
            i, code = ln.strip().split(","); i=int(i)-1
            idxs.append(i)
            letter = code.split(".")[1][0] if "." in code and len(code.split(".")[1])>0 else (code[0] if code else "")
            nuc[i] = norm_base(letter)
    return nuc, (max(idxs) if idxs else -1)

onehot = {"A":[1,0,0,0], "C":[0,1,0,0], "G":[0,0,1,0], "U":[0,0,0,1], "N":[0,0,0,0]}

def norm_base(x: str) -> str:
    if not x: return "N"
    b = x.upper()
    if b == "T": return "U"
    return b if b in ("A","C","G","U") else "N"

def canonical_pairs_from_cmt(cmt):
    ic,jc=np.where(cmt==1)
    return {(min(i,j),max(i,j)) for i,j in zip(ic,jc) if i!=j}

def cross_window_pairs(canonical,N,W):
    cand=set()
    for x,y in canonical:
        for i in range(max(0,x-W),min(N,x+W+1)):
            for j in range(max(0,y-W),min(N,y+W+1)):
                if i<j: cand.add((i,j))
    return cand

def sample_negatives_local_global(noncan,cmt,neigh,N,n_pos,neg_per_pos,local_frac,W,rng):
    canon=canonical_pairs_from_cmt(cmt)
    canon_set=set(canon)
    local_cand=cross_window_pairs(canon,N,W) if canon else set()
    def valid(i,j):
        if i>=j or (i,j) in canon_set or neigh[i,j] or noncan[i,j]: return False
        return True
    local_pool=[p for p in local_cand if valid(*p)]
    global_pool=[(i,j) for i in range(N) for j in range(i+1,N)
                 if (i,j) not in local_cand and valid(i,j)]
    m=n_pos*neg_per_pos
    n_local=min(int(round(m*local_frac)),len(local_pool))
    n_global=min(m-n_local,len(global_pool))
    local_negs=rng.choice(local_pool,n_local,replace=False).tolist() if n_local else []
    global_negs=rng.choice(global_pool,n_global,replace=False).tolist() if n_global else []
    return local_negs+global_negs

def load_dataset(data_root,W,neg_per_pos,neg_local_frac,seed):
    rng=np.random.default_rng(seed)
    paths={k:os.path.join(data_root,k) for k in ("amt","cmt","idx")}
    graphs=[]; edge_lists=[]; labels=[]; seqs=[]
    cmt_mats=[]; neigh_mats=[]; amt_mats=[]; cand_negs=[]
    for fname in os.listdir(paths["amt"]):
        if not fname.endswith(".amt"): continue
        base=fname[:-4]
        p_amt=paths["amt"]+"/"+fname
        p_cmt=paths["cmt"]+f"/{base}.cmt"
        p_idx=paths["idx"]+f"/{base}.idx"
        if not (os.path.exists(p_cmt) and os.path.exists(p_idx)): continue
        cmt=load_matrix(p_cmt); amt=load_matrix(p_amt)
        noncan=(amt>1).astype(int); neigh=(amt==-1).astype(int)
        nuc_map,idx_max=load_idx(p_idx)
        N=max(idx_max+1,cmt.shape[0])
        pad=lambda m: m if m.shape[0]>=N else np.pad(m,((0,N-m.shape[0]),(0,N-m.shape[0])))
        cmt,noncan,neigh,amt=(pad(m) for m in (cmt,noncan,neigh,amt))
        # graph
        src,dst,efeat=[],[],[]
        def add(mat,v):
            ii,jj=np.where(mat==1)
            src.extend(ii.tolist()); dst.extend(jj.tolist()); efeat.extend([v]*len(ii))
            src.extend(jj.tolist()); dst.extend(ii.tolist()); efeat.extend([v]*len(ii))
        add(cmt,[1,0]); add(neigh,[0,1])
        g=dgl.graph((src,dst),num_nodes=N)
        g.edata["feat"]=torch.tensor(efeat,dtype=torch.float32)
        g.ndata["feat"]=torch.tensor([onehot.get(nuc_map.get(i),[0,0,0,0]) for i in range(N)],dtype=torch.float32)
        g.ndata["pos"]=torch.arange(N); g.ndata["len"]=torch.full((N,),N)
        # positives i wstępne negatywy
        pos=[(i,j) for i in range(N) for j in range(i+1,N) if noncan[i,j]]
        if not pos: continue
        cand_neg=[(i,j) for i in range(N) for j in range(i+1,N)
                  if not noncan[i,j] and not neigh[i,j] and not cmt[i,j]]
        init_neg=sample_negatives_local_global(noncan,cmt,neigh,N,len(pos),neg_per_pos,neg_local_frac,W,rng)
        pairs=torch.tensor(pos+init_neg,dtype=torch.long)
        labs =torch.tensor([1]*len(pos)+[0]*len(init_neg),dtype=torch.float32)
        graphs.append(g); edge_lists.append(pairs); labels.append(labs); seqs.append("".join(nuc_map.get(i,"N") for i in range(N)))
        cmt_mats.append(cmt); neigh_mats.append(neigh); amt_mats.append(amt); cand_negs.append(cand_neg)
    return graphs,edge_lists,labels,seqs,cmt_mats,neigh_mats,amt_mats,cand_negs

# Positional encoding
class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
    def forward(self, T: int, device):
        position = torch.arange(T, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.dim, 2, device=device) * (-math.log(10000.0) / self.dim))
        pe = torch.zeros(T, self.dim, device=device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

# Transformer (SSM)
class SequenceStructureModule(nn.Module):
    def __init__(self, dim: int, n_layers: int = 2, nhead: int = 8, dropout: float = 0.1,
                 use_posenc: bool = True):
        super().__init__()
        self.use_posenc = use_posenc
        self.posenc = SinusoidalPositionalEncoding(dim) if use_posenc else None
        self.fuse = nn.Sequential(
            nn.Linear(2*dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )
        enc_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=nhead, dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers, norm=nn.LayerNorm(dim))

    def forward(self, seq_emb, struct_emb, key_padding_mask=None):
        B, T, D = seq_emb.shape
        device = seq_emb.device
        if self.use_posenc:
            pe = self.posenc(T, device)
            seq_emb = seq_emb + pe
            struct_emb = struct_emb + pe
        x = torch.cat([seq_emb, struct_emb], dim=2)
        x = self.fuse(x)
        out = self.encoder(x, mask=None, src_key_padding_mask=key_padding_mask)
        return out


# Etap 1: Bi-encoder
class ResidueEncoder(nn.Module):
    def __init__(self, h=256):
        super().__init__()
        self.rinalmo, self.alpha = get_pretrained_model("giga-v1")
        self.rinalmo.eval()
        for p in self.rinalmo.parameters(): p.requires_grad_(False)
        self.proj = nn.Sequential(nn.Linear(1280, h), nn.ReLU(), nn.Linear(h, h))
        self.bn   = nn.BatchNorm1d(h)

    @torch.no_grad()
    def _enc_raw(self, seqs, device):
        toks = torch.tensor(self.alpha.batch_tokenize(seqs), dtype=torch.int64, device=device)
        with torch.cuda.amp.autocast(enabled=(device.type=="cuda")):
            rep = self.rinalmo(toks)["representation"]  # (B,1+T+1,1280)
        outs = [rep[i,1:1+len(s)] for i,s in enumerate(seqs)]
        return torch.cat(outs,0)

    def forward(self, seqs, device):
        X = self._enc_raw(seqs, device)
        X = self.proj(X)
        X = self.bn(X).relu()
        return X

class PairBiEncoder(nn.Module):
    def __init__(self, h=256, bucket_bins=(15,30,50,100,200), use_log_dist=True):
        super().__init__()
        self.proj = nn.Linear(h, h, bias=False)
        self.bucket_bins = torch.as_tensor(bucket_bins, dtype=torch.float32)
        self.bucket_bias = nn.Embedding(len(bucket_bins)+1, 1)
        self.use_log_dist = use_log_dist
        self.wd = nn.Parameter(torch.tensor(0.0))

    def pair_score(self, U, pos, glen):
        def _score_pairs(pairs):
            s = pairs[:,0]; d = pairs[:,1]
            Ui = self.proj(U[s]); Uj = self.proj(U[d])
            logits = (Ui * Uj).sum(-1, keepdim=True)
            dist = (pos[d]-pos[s]).abs().unsqueeze(1).float()
            bucket = torch.bucketize(dist.squeeze(1), self.bucket_bins.to(U.device), right=False)
            logits = logits + self.bucket_bias(bucket)
            if self.use_log_dist:
                logits = logits + self.wd * torch.log1p(dist)
            return logits.squeeze(1)
        return _score_pairs

@torch.no_grad()
def bienc_topk_candidates(res_enc, bienc, g, seq, window, cmt, neigh, topk, device, batch_cap=250000):
    if topk <= 0:
        return []
    N = g.num_nodes()
    U = res_enc([seq], device)
    pos = g.ndata["pos"].to(device).float()
    glen = g.ndata["len"].to(device).float()
    score_pairs = bienc.pair_score(U, pos, glen)

    canon = canonical_pairs_from_cmt(cmt)
    cand  = cross_window_pairs(canon, N, window) if canon else set()
    def valid(i,j):
        if i>=j or cmt[i,j]==1 or neigh[i,j]==1: return False
        return True
    cand = [p for p in cand if valid(*p)]
    if not cand: return []

    cand_t = torch.tensor(cand, dtype=torch.long, device=device)
    scores = []
    for s in range(0, len(cand_t), batch_cap):
        scores.append(torch.sigmoid(score_pairs(cand_t[s:s+batch_cap])).float().cpu())
    scores = torch.cat(scores)
    idx = torch.topk(scores, k=min(topk, len(scores)))[1].numpy()
    return [cand[i] for i in idx]


# Etap 2: Reranker GNN 
class GNN(nn.Module):
    def __init__(self,h=128,edge_dim=2,depth=4,bucket_bins=(20,50,100),drop=0.15,drop_pred=0.25,
                 use_log_dist=False, use_ssm: bool=False, ssm_layers: int=2, ssm_heads: int=8,
                 ssm_dropout: float=0.1, ssm_no_posenc: bool=False, use_layernorm: bool=False,
                 gnn_type: str="nnconv", gat_heads: int=4, gat_feat_drop: float=0.0,
                 gat_attn_drop: float=0.0, gat_residual: bool=False):
        super().__init__()
        self.use_log_dist=use_log_dist
        self.bucket_bins=torch.as_tensor(bucket_bins,dtype=torch.float32)
        self.gnn_type = gnn_type
        self.gat_heads = gat_heads

        # Limity chunkowania RiNALMo
        self.rinalmo_max_len = 1024
        self.rinalmo_overlap = 64

        # RINAlMo encoder 
        self.rinalmo,self.alpha=get_pretrained_model("giga-v1"); self.rinalmo.eval()
        for p in self.rinalmo.parameters(): p.requires_grad_(False)

        # Projekcja sekwencji
        self.seq_proj=nn.Sequential(nn.Linear(1280,h),nn.ReLU(),nn.Linear(h,h),nn.ReLU())
        Norm1D = (lambda dim: nn.LayerNorm(dim)) if use_layernorm else (lambda dim: nn.BatchNorm1d(dim))
        self.bn0=Norm1D(h)

        # Blok GNN 
        self.drop=nn.Dropout(drop)

        if gnn_type == "nnconv":
            edge_net=nn.Sequential(nn.Linear(edge_dim,h*h),nn.ReLU(),nn.BatchNorm1d(h*h))
            self.gconvs=nn.ModuleList([NNConv(h,h,edge_net,"mean") for _ in range(depth)])
            self.bns   =nn.ModuleList([Norm1D(h) for _ in range(depth)])
        elif gnn_type == "gat":
            assert h % gat_heads == 0, "hidden (h) must be divisible by gat_heads"
            out_per_head = h // gat_heads
            self.gconvs = nn.ModuleList([
                GATConv(in_feats=h, out_feats=out_per_head, num_heads=gat_heads,
                        feat_drop=gat_feat_drop, attn_drop=gat_attn_drop,
                        residual=gat_residual, allow_zero_in_degree=True)
                for _ in range(depth)
            ])
            self.bns = nn.ModuleList([Norm1D(h) for _ in range(depth)])
        elif gnn_type == "hybrid":
            assert h % gat_heads == 0, "hidden (h) must be divisible by gat_heads"
            out_per_head = h // gat_heads
            self.gconvs = nn.ModuleList()
            self.bns    = nn.ModuleList()
            for layer in range(depth):
                if layer % 2 == 0:
                    edge_net=nn.Sequential(nn.Linear(edge_dim,h*h),nn.ReLU(),nn.BatchNorm1d(h*h))
                    self.gconvs.append(NNConv(h,h,edge_net,"mean"))
                else:
                    self.gconvs.append(
                        GATConv(in_feats=h, out_feats=out_per_head, num_heads=gat_heads,
                                feat_drop=gat_feat_drop, attn_drop=gat_attn_drop,
                                residual=gat_residual, allow_zero_in_degree=True)
                    )
                self.bns.append(Norm1D(h))
        else:
            raise ValueError(f"Unknown gnn_type={gnn_type}")

        # Transformer (opcjonalny SSM)
        self.use_ssm = use_ssm
        if use_ssm:
            self.ssm = SequenceStructureModule(
                h, n_layers=ssm_layers, nhead=ssm_heads, dropout=ssm_dropout,
                use_posenc=not ssm_no_posenc
            )

        # Head do predykcji par
        self.bucket_embed=nn.Embedding(len(bucket_bins)+1,h//4)
        in_dim=2*h+1+(h//4)+(1 if use_log_dist else 0)
        self.pred=nn.Sequential(nn.Linear(in_dim,h),nn.ReLU(),nn.Dropout(drop_pred),nn.Linear(h,1))

    # pakowanie/rozpakowanie (sum_T,h) <-> (B,Tmax,h)
    def _pack_to_padded(self, X_cat, seqs):
        lens = [len(s) for s in seqs]
        B, Tm, D = len(lens), max(lens), X_cat.size(-1)
        device = X_cat.device
        padded = torch.zeros(B, Tm, D, device=device)
        pad_mask = torch.ones(B, Tm, dtype=torch.bool, device=device)
        offs = 0
        for i, L in enumerate(lens):
            padded[i, :L] = X_cat[offs:offs+L]
            pad_mask[i, :L] = False
            offs += L
        return padded, pad_mask, lens

    def _unpack_from_padded(self, X_padded, lens):
        chunks = [X_padded[i, :L] for i, L in enumerate(lens)]
        return torch.cat(chunks, dim=0)

    def _enc_one_seq_chunked(self, seq: str, device, max_len: int, overlap: int):
        
        # Długość docelowa to liczba nukleotydów (bez BOS/EOS)
        T_total = len(seq)
        # Prosty przypadek — bez chunkowania
        toks = torch.tensor([self.alpha.batch_tokenize([seq])[0]], dtype=torch.int64, device=device)
        if T_total <= max_len:
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=(device.type=="cuda")):
                rep = self.rinalmo(toks)["representation"]  # (1,1+T+1,1280)
            x = rep[0, 1:1+T_total]                         # (T,1280)
            return self.seq_proj(x)

        # Chunkowanie z usuwaniem duplikatów
        pieces = []
        start = 0
        prev_end = 0
        while start < T_total:
            end = min(start + max_len, T_total)
            sub_seq = seq[start:end]
            sub_toks = torch.tensor([self.alpha.batch_tokenize([sub_seq])[0]],
                                    dtype=torch.int64, device=device)
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=(device.type=="cuda")):
                sub_rep = self.rinalmo(sub_toks)["representation"]  # (1,1+len+1,1280)
            sub_x = sub_rep[0, 1:-1]  # (len, 1280) odpowiada pozycjom [start, end)

            new_from = max(prev_end - start, 0)
            if new_from < sub_x.size(0):
                pieces.append(sub_x[new_from:])

            prev_end = end
            if end == T_total:
                break
            start = end - overlap  # przesunięcie 

        x_full = torch.cat(pieces, dim=0)  
        # Projekcja do h
        x_full = self.seq_proj(x_full)
        if x_full.size(0) != T_total:
            print(f"[WARN] RiNALMo chunk stitch mismatch: got {x_full.size(0)} vs {T_total} — fixing by slicing/padding.")
            if x_full.size(0) > T_total:
                x_full = x_full[:T_total]
            else:
                pad = torch.zeros(T_total - x_full.size(0), x_full.size(1), device=device, dtype=x_full.dtype)
                x_full = torch.cat([x_full, pad], dim=0)
        return x_full

    def _enc(self, seqs, device):

        max_len = getattr(self, "rinalmo_max_len", 1024)
        overlap = getattr(self, "rinalmo_overlap", 64)

        outs = []
        for s in seqs:
            outs.append(self._enc_one_seq_chunked(s, device, max_len, overlap))
        return torch.cat(outs, dim=0)  # (sum_T, h)

    def _align_to_graph_nodes(self, g, x):

        N = g.num_nodes()
        if x.size(0) == N:
            return x
        print(f"[WARN] node/feature length mismatch: x={x.size(0)} vs N={N} — aligning.")
        if x.size(0) > N:
            return x[:N]
        pad = torch.zeros(N - x.size(0), x.size(1), device=x.device, dtype=x.dtype)
        return torch.cat([x, pad], dim=0)

    def _gat_forward_layer(self, layer, g, x):
        x = self.gconvs[layer](g, x)
        if isinstance(x, tuple):
            x = x[0]
        if x.dim() == 3:
            x = x.reshape(x.shape[0], -1)
        return x

    def forward(self,g,pairs,seqs):
        device=pairs.device

        # sekwencja (RiNALMo) -> proj (w _enc_one_seq_chunked) -> dopasowanie -> Norm -> ReLU
        seq_tok=self._enc(seqs,device)           # (sum_T, h)
        seq_tok=self._align_to_graph_nodes(g, seq_tok)
        assert seq_tok.size(0) == g.num_nodes(), f"Features ({seq_tok.size(0)}) != nodes ({g.num_nodes()})"
        seq_tok=self.bn0(seq_tok).relu()

        # Encoder grafowy
        x=seq_tok
        for li, (conv, bn) in enumerate(zip(self.gconvs, self.bns)):
            if self.gnn_type == "nnconv":
                x = x + self.drop(bn(conv(g, x, g.edata["feat"]))).relu()
            elif self.gnn_type == "gat":
                x = x + self.drop(bn(self._gat_forward_layer(li, g, x))).relu()
            elif self.gnn_type == "hybrid":
                if isinstance(conv, NNConv):
                    x = x + self.drop(bn(conv(g, x, g.edata["feat"]))).relu()
                else:
                    x = x + self.drop(bn(self._gat_forward_layer(li, g, x))).relu()

        # Transformer (SSM)
        if self.use_ssm:
            seq_pad, pad_mask, lens = self._pack_to_padded(seq_tok, seqs)
            str_pad, _       , _    = self._pack_to_padded(x,       seqs)
            with torch.cuda.amp.autocast(enabled=False):
                fused = self.ssm(seq_pad.float(), str_pad.float(), key_padding_mask=pad_mask)
            fused = self._unpack_from_padded(fused, lens)
            x = x + fused.to(x.dtype)

        # predykcja par
        s,d=pairs[:,0],pairs[:,1]
        pos=g.ndata["pos"].to(device).float(); glen=g.ndata["len"].to(device).float()
        dist=(pos[d]-pos[s]).abs().unsqueeze(1); dist_norm=dist/glen[s].unsqueeze(1)
        bucket=torch.bucketize(dist.squeeze(1),self.bucket_bins.to(device),right=False)
        feats=[x[s],x[d],dist_norm,self.bucket_embed(bucket)]
        if self.use_log_dist: feats.append(torch.log1p(dist))
        out = self.pred(torch.cat(feats,1)).squeeze(-1)
        out = torch.nan_to_num(out, nan=0.0, posinf=1e4, neginf=-1e4)
        return out

# Data / collate
@torch.no_grad()
def hard_mine_graph(model,g,seq,cand_neg,pos_cnt,topk,device,batch_cap=200000):
    model.eval(); scores=[]
    cand=torch.as_tensor(cand_neg,dtype=torch.long,device=device)
    for s in range(0,len(cand),batch_cap):
        logits=model(g,cand[s:s+batch_cap],[seq])
        scores.append(torch.sigmoid(logits).cpu())
    scores=torch.cat(scores)
    idx=torch.topk(scores,k=min(len(scores),topk*pos_cnt))[1].numpy()
    return [cand[i] for i in idx]

class GraphTrainDS(torch.utils.data.Dataset):
    def __init__(self,g,e,l,s): self.g,self.e,self.l,self.s=g,e,l,s
    def __len__(self): return len(self.g)
    def __getitem__(self,i): return self.g[i],self.e[i],self.l[i],self.s[i]

def collate(batch):
    Gs,Es,Ls,Ss=zip(*batch); bg=dgl.batch(list(Gs))
    offs=np.add.accumulate([0]+[g.num_nodes() for g in Gs[:-1]])
    Es_shift=[e+o for e,o in zip(Es,offs)]
    return bg,torch.cat(Es_shift),torch.cat(Ls),list(Ss)

# Ewaluacje
def evaluate_sampled(model,loader,device,thr=0.5):
    model.eval(); logits,labs=[],[]
    with torch.no_grad():
        for g,e,l,s in loader:
            g,e,l=g.to(device),e.to(device),l.to(device)
            with torch.cuda.amp.autocast(enabled=device.type=="cuda"):
                logits.append(model(g,e,s).cpu()); labs.append(l.cpu())
    logits=torch.cat(logits); labs=torch.cat(labs).numpy(); probs=torch.sigmoid(logits).numpy()
    if not np.isfinite(probs).all():
        bad = (~np.isfinite(probs)).sum()
        print(f"[WARN] non-finite probs in sampled eval: {bad} -> cleaning")
        probs = np.nan_to_num(probs, nan=0.5, posinf=1.0, neginf=0.0)
    pc,rc,thr_grid=precision_recall_curve(labs,probs); pr_auc=auc(rc,pc)
    preds=(probs>=thr).astype(int)
    P,R,F1=precision_score(labs,preds,zero_division=0),recall_score(labs,preds,zero_division=0),f1_score(labs,preds,zero_division=0)
    return P,R,F1,pr_auc,probs,labs,thr_grid,pc,rc

def enumerate_window_candidates(noncan,cmt,neigh,N,W):
    canon=canonical_pairs_from_cmt(cmt); cand=cross_window_pairs(canon,N,W) if canon else set()
    pairs,labs=[],[]
    for i,j in sorted(cand):
        if cmt[i,j]==1 or neigh[i,j]==1: continue
        pairs.append((i,j)); labs.append(1 if noncan[i,j] else 0)
    return (None,None) if not pairs else (np.array(pairs),np.array(labs,dtype=np.float32))

def evaluate_windowed(model,device,graphs,seqs,cmt_mats,neigh_mats,amt_mats,idxs,W,
                      res_enc=None,bienc=None,topk=0,batch_cap=100000):
    model.eval(); logits,labs=[],[]
    with torch.no_grad():
        for k in idxs:
            g=graphs[k].to(device); seq=seqs[k]
            noncan=(amt_mats[k]>1).astype(int); N=g.num_nodes()
            if topk>0 and (res_enc is not None) and (bienc is not None):
                top_pairs = bienc_topk_candidates(res_enc, bienc, g, seq, W, cmt_mats[k], neigh_mats[k], topk, device)
                if not top_pairs: 
                    cand_np,lab_np = enumerate_window_candidates(noncan,cmt_mats[k],neigh_mats[k],N,W)
                else:
                    labs_np = np.array([1 if noncan[i,j] else 0 for (i,j) in top_pairs], dtype=np.float32)
                    cand_np,lab_np = np.array(top_pairs, dtype=np.int64), labs_np
            else:
                cand_np,lab_np=enumerate_window_candidates(noncan,cmt_mats[k],neigh_mats[k],N,W)
            if cand_np is None: continue
            cand=torch.from_numpy(cand_np).to(device); lab=torch.from_numpy(lab_np).to(device)
            for s in range(0,len(cand),batch_cap):
                logits.append(model(g,cand[s:s+batch_cap],[seq]).cpu())
                labs.append(lab[s:s+batch_cap].cpu())
    if not logits: return None
    logits=torch.cat(logits); labs=torch.cat(labs).numpy(); probs=torch.sigmoid(logits).numpy()
    if not np.isfinite(probs).all():
        bad = (~np.isfinite(probs)).sum()
        print(f"[WARN] non-finite probs in windowed eval: {bad} -> cleaning")
        probs = np.nan_to_num(probs, nan=0.5, posinf=1.0, neginf=0.0)
    pc,rc,thr_grid=precision_recall_curve(labs,probs); pr_auc=auc(rc,pc)
    preds=(probs>=0.5).astype(int)
    P,R,F1=precision_score(labs,preds,zero_division=0),recall_score(labs,preds,zero_division=0),f1_score(labs,preds,zero_division=0)
    return P,R,F1,pr_auc,probs,labs,thr_grid,pc,rc

def best_threshold_from_pr(pc,rc,thr_grid,eps=1e-9):
    f1=(2*pc*rc)/(pc+rc+eps); thr_full=np.concatenate([thr_grid,[1.0]])
    idx=int(np.nanargmax(f1)); return float(thr_full[idx]),float(f1[idx])


# Lossy i wagi
class SmoothedBCEWithLogits(nn.Module):
    def __init__(self, pos_weight=None, eps=0.05, reduction="mean"):
        super().__init__()
        self.eps = eps
        self.reduction = reduction
        self.base = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")
    def forward(self, logits, targets):
        t = targets * (1.0 - self.eps) + 0.5 * self.eps
        loss = self.base(logits, t)
        return loss.mean() if self.reduction=="mean" else loss.sum()

def effective_pos_weight(num_pos, num_neg, beta=0.999):
    n_pos = float(num_pos); n_neg = float(num_neg)
    alpha_pos = (1 - beta) / (1 - pow(beta, max(n_pos, 1.0)))
    alpha_neg = (1 - beta) / (1 - pow(beta, max(n_neg, 1.0)))
    return torch.tensor([alpha_pos / alpha_neg], dtype=torch.float32)


# Trening bi-encodera (lekki)
class BiEncBCETrainer:
    def __init__(self, res_enc, bienc, lr=5e-4, wd=1e-5, eps=0.05):
        self.res_enc = res_enc
        self.bienc   = bienc
        params = list(bienc.parameters()) + list(res_enc.proj.parameters()) + list(res_enc.bn.parameters())
        self.opt = torch.optim.AdamW(params, lr=lr, weight_decay=wd)
        self.loss = SmoothedBCEWithLogits(pos_weight=None, eps=eps)

    def step(self, g, seq, pos_pairs, neg_pairs, device, batch=200000):
        self.bienc.train(); self.res_enc.train()
        U = self.res_enc([seq], device)
        pos_v = g.ndata["pos"].to(device).float()
        glen  = g.ndata["len"].to(device).float()
        scorer = self.bienc.pair_score(U, pos_v, glen)

        pairs = torch.tensor(pos_pairs+neg_pairs, dtype=torch.long, device=device)
        labs  = torch.tensor([1]*len(pos_pairs) + [0]*len(neg_pairs), dtype=torch.float32, device=device)
        self.opt.zero_grad(set_to_none=True)
        for s in range(0, len(pairs), batch):
            logits = scorer(pairs[s:s+batch])
            loss = self.loss(logits, labs[s:s+batch])
            loss.backward()
        torch.nn.utils.clip_grad_norm_(list(self.bienc.parameters())+list(self.res_enc.parameters()), max_norm=1.0)
        self.opt.step()
        return 0.0

def save_stage1(res_enc, bienc, path):
    torch.save({
        "res_proj": res_enc.proj.state_dict(),
        "res_bn":   res_enc.bn.state_dict(),
        "bienc":    bienc.state_dict(),
    }, path)
    print("⇒ saved Stage-1 to", path)

def load_stage1(res_enc, bienc, path, map_location=None):
    ckpt = torch.load(path, map_location=map_location)
    res_enc.proj.load_state_dict(ckpt["res_proj"])
    res_enc.bn.load_state_dict(ckpt["res_bn"])
    bienc.load_state_dict(ckpt["bienc"])
    print(" loaded Stage-1 from", path)


# Trening główny
def train_main(args):
    set_seed(args.seed)
    DEVICE=torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    BUCKET_BINS=tuple(int(x) for x in args.bins.split(","))
    (GRAPHS,EDGE_LISTS,LABELS,SEQS,
     CMT_MATS,NEIGH_MATS,AMT_MATS,ALL_CAND_NEG)=load_dataset(args.data_root,args.window,
                                                             args.neg_per_pos,args.neg_local_frac,args.seed)
    FULL=GraphTrainDS(GRAPHS,EDGE_LISTS,LABELS,SEQS)
    ids=list(range(len(FULL)))
    if len(ids) >= 3:
        tr,tmp=train_test_split(ids,test_size=0.3,random_state=args.seed)
        va,te=train_test_split(tmp,test_size=0.5,random_state=args.seed)
    else:
        tr,va,te=ids[:1],ids[1:2],ids[2:3]

    mk=lambda idx,sh: DataLoader(torch.utils.data.Subset(FULL,idx),batch_size=args.batch_size,
                                 shuffle=sh,collate_fn=collate,pin_memory=DEVICE.type=="cuda")
    train_loader,val_loader,test_loader=mk(tr,True),mk(va,False),mk(te,False)

    num_pos = sum(l.sum().item() for l in LABELS)
    num_all = sum(len(l) for l in LABELS)
    num_neg = num_all - num_pos
    if args.cb_posw:
        pos_w = effective_pos_weight(num_pos, num_neg, beta=0.999).to(DEVICE)
    else:
        pos_ratio = num_pos / max(1.0, num_all)
        pos_w=torch.tensor([(1-pos_ratio)/max(pos_ratio,1e-8)*1.2],device=DEVICE)

    # MODELE
    res_enc = ResidueEncoder(h=args.hidden).to(DEVICE)
    bienc   = PairBiEncoder(h=args.hidden, bucket_bins=BUCKET_BINS, use_log_dist=args.use_log_dist).to(DEVICE)

    if args.load_stage1:
        load_stage1(res_enc, bienc, args.load_stage1, map_location=DEVICE)
        args.stage1_epochs = 0

    if args.stage1_epochs > 0:
        print(f"[Stage1] Trening bi-encoder: {args.stage1_epochs} epok …")
        trainer = BiEncBCETrainer(res_enc, bienc, lr=args.stage1_lr, wd=1e-5, eps=args.label_smoothing)
        rng=np.random.default_rng(args.seed)
        for ep in range(1, args.stage1_epochs+1):
            tot=0.0; steps=0
            for k in tr:
                g=GRAPHS[k].to(DEVICE); seq=SEQS[k]
                noncan=(AMT_MATS[k]>1).astype(int); N=g.num_nodes()
                pos_pairs=[(i,j) for i in range(N) for j in range(i+1,N) if noncan[i,j]]
                if not pos_pairs: continue
                neg_pairs=sample_negatives_local_global(noncan,CMT_MATS[k],NEIGH_MATS[k],N,len(pos_pairs),
                                                        args.stage1_neg_per_pos, args.neg_local_frac, args.window, rng)
                trainer.step(g, seq, pos_pairs, neg_pairs, DEVICE)
                tot += 1; steps += 1
            print(f"[Stage1] ep={ep}  steps={steps}")
        print("[Stage1] done.\n")
    
    if args.stage1_epochs > 0 and args.save_stage1:
        save_stage1(res_enc, bienc, args.save_stage1)

    model=GNN(h=args.hidden,depth=args.depth,bucket_bins=BUCKET_BINS,
              drop=args.dropout,drop_pred=args.dropout_pred,use_log_dist=args.use_log_dist,
              use_ssm=args.use_ssm, ssm_layers=args.ssm_layers, ssm_heads=args.ssm_heads,
              ssm_dropout=args.ssm_dropout, ssm_no_posenc=args.ssm_no_posenc,
              use_layernorm=args.use_layernorm,
              gnn_type=args.gnn_type, gat_heads=args.gat_heads,
              gat_feat_drop=args.gat_feat_drop, gat_attn_drop=args.gat_attn_drop,
              gat_residual=args.gat_residual).to(DEVICE)
    model.rinalmo_max_len = int(getattr(args, "rinalmo_max_len", 1024))
    model.rinalmo_overlap = int(getattr(args, "rinalmo_overlap", 64))

    loss_fn=SmoothedBCEWithLogits(pos_weight=pos_w, eps=args.label_smoothing)
    opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-5, eps=args.adam_eps)
    scaler=torch.cuda.amp.GradScaler(enabled=(DEVICE.type=="cuda" and not args.no_mixed))
    sched=torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt,T_0=4,T_mult=2,eta_min=1e-5)
    BEST,best_state,noimp=-math.inf,None,0

    for ep in range(1,args.epochs+1):
        model.train(); tot=0.0
        for g,e,l,s in train_loader:
            g,e,l=g.to(DEVICE),e.to(DEVICE),l.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=DEVICE.type=="cuda" and not args.no_mixed):
                loss=loss_fn(model(g,e,s),l)
            scaler.scale(loss).backward()
            if DEVICE.type=="cuda" and not args.no_mixed:
                scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip_grad)
            scaler.step(opt); scaler.update(); tot+=loss.item()
        sched.step(ep)

        P,R,F1,auc_pr,_,_,thr_grid,pc,rc=evaluate_sampled(model,val_loader,DEVICE)
        print(f"E{ep:03d}  loss={tot/len(train_loader):.4f}  PR-AUC={auc_pr:.3f}  P={P:.3f} R={R:.3f} F1={F1:.3f}")

        if auc_pr>BEST:
            BEST,noimp=auc_pr,0
            best_state={"state":{k:v.cpu() for k,v in model.state_dict().items()},
                        "thr_grid":thr_grid,"pc":pc,"rc":rc}
        else:
            noimp+=1
        if noimp>=args.patience:
            print("[EARLY STOP]"); break

    model.load_state_dict(best_state["state"])
    print("\n[INFO] windowed validation …")
    out=evaluate_windowed(model,DEVICE,GRAPHS,SEQS,CMT_MATS,NEIGH_MATS,AMT_MATS,va,args.window,
                          res_enc=res_enc,bienc=bienc,topk=args.stage1_topk)
    if out:
        Pvw,Rvw,F1vw,auc_vw,probs,labs,thr_grid,pc,rc=out
        thr_best,_=best_threshold_from_pr(pc,rc,thr_grid)
        print(f"[VAL-window] PR-AUC={auc_vw:.3f}  Best-thr={thr_best:.4f}")
    else:
        thr_best=0.5
        auc_vw=None

    print("\n[INFO] windowed test …")
    out=evaluate_windowed(model,DEVICE,GRAPHS,SEQS,CMT_MATS,NEIGH_MATS,AMT_MATS,te,args.window,
                          res_enc=res_enc,bienc=bienc,topk=args.stage1_topk)
    if out:
        Pte0,Rte0,F1te0,auc_te,probs,labs,*_=out
        preds=(probs>=thr_best).astype(int)
        Pte,Rte,F1te=precision_score(labs,preds,zero_division=0),recall_score(labs,preds,zero_division=0),f1_score(labs,preds,zero_division=0)
        print(f"[TEST-window] PR-AUC={auc_te:.3f}  P={Pte:.3f} R={Rte:.3f} F1={F1te:.3f}")

    torch.save({"state":best_state["state"],"threshold":thr_best,"args":vars(args)},args.save)
    print("⇒ saved",args.save)

    return float(auc_vw if auc_vw is not None else BEST)


if __name__=="__main__":
    args=build_argparser().parse_args()
    _ = train_main(args)
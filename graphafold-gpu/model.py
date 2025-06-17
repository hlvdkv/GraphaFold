import os, random, typing as tp, math, multiprocessing as mp
import numpy as np, torch, dgl
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    precision_recall_curve, auc,
)

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %% I/O helpers

def load_matrix(p: str) -> np.ndarray:
    return np.loadtxt(p, delimiter=",")


def load_idx(path: str) -> tp.Tuple[dict[int, str | None], int]:
    nuc, idxs = {}, []
    with open(path) as fh:
        for ln in fh:
            if not ln.strip():
                continue
            i, code = ln.strip().split(",")
            i = int(i) - 1; idxs.append(i)
            nuc[i] = code.split(".")[1][0] if "." in code else None
    return nuc, (max(idxs) if idxs else -1)


# %% Dataset construction 
TRAIN_FOLDER, CMT_FOLDER, IDX_FOLDER = "TrainingSet/amt", "TrainingSet/cmt", "TrainingSet/idx"
GRAPHS, EDGE_LISTS, LABELS = [], [], []

for fname in os.listdir(TRAIN_FOLDER):
    if not fname.endswith(".amt"):
        continue
    base = fname[:-4]
    paths = {
        "amt": os.path.join(TRAIN_FOLDER, fname),
        "cmt": os.path.join(CMT_FOLDER, f"{base}.cmt"),
        "idx": os.path.join(IDX_FOLDER, f"{base}.idx"),
    }
    if not all(map(os.path.exists, paths.values())):
        print("[WARN] missing companion file for", base); continue

    cmt, amt = load_matrix(paths["cmt"]), load_matrix(paths["amt"])
    noncan, neigh = (amt > 1).astype(int), (amt == -1).astype(int)
    nuc_map, idx_max = load_idx(paths["idx"]); N = max(idx_max + 1, cmt.shape[0])

    def _pad(mat: np.ndarray, size: int):
        if mat.shape[0] >= size: return mat
        out = np.zeros((size, size), dtype=mat.dtype)
        out[: mat.shape[0], : mat.shape[1]] = mat; return out

    cmt, noncan, neigh = (_pad(m, N) for m in (cmt, noncan, neigh))

    # Build graph
    src, dst, efeat = [], [], []
    def _add_edges(mat: np.ndarray, vec):
        i, j = np.where(mat == 1)
        src.extend(i.tolist()); dst.extend(j.tolist()); efeat.extend([vec]*len(i))
        src.extend(j.tolist()); dst.extend(i.tolist()); efeat.extend([vec]*len(i))
    _add_edges(noncan, [1,0,0]); _add_edges(cmt, [0,1,0]); _add_edges(neigh,[0,0,1])

    edge_max = max(src+dst) if src else -1
    N = max(idx_max+1, edge_max+1)
    cmt, noncan, neigh = (_pad(m, N) for m in (cmt, noncan, neigh))

    g = dgl.graph((src, dst), num_nodes=N)
    g.edata['feat'] = torch.as_tensor(efeat, dtype=torch.float32)
    onehot = {'A':[1,0,0,0],'C':[0,1,0,0],'G':[0,0,1,0],'U':[0,0,0,1],None:[0,0,0,0]}
    g.ndata['feat'] = torch.as_tensor([onehot[nuc_map.get(i)] for i in range(N)], dtype=torch.float32)

    pos = [(i,j) for i in range(N) for j in range(i+1,N) if noncan[i,j]]
    if not pos: continue

    neg, K = [], 10
    for i in range(N):
        excl = {j for u,j in pos if u==i} | set(np.where(noncan[i])[0]) | set(np.where(cmt[i])[0]) | {i}
        pool = [j for j in range(i+1,N) if j not in excl]
        if pool:
            neg.extend([(i,j) for j in np.random.choice(pool, min(K,len(pool)), replace=False)])

    GRAPHS.append(g)
    EDGE_LISTS.append(torch.tensor(pos+neg, dtype=torch.long))
    LABELS.append(torch.tensor([1]*len(pos)+[0]*len(neg), dtype=torch.float32))


# %% DataLoader
class GraphDS(torch.utils.data.Dataset):
    def __init__(self,G,E,L): self.G,self.E,self.L = G,E,L
    def __len__(self): return len(self.G)
    def __getitem__(self,idx): return self.G[idx],self.E[idx],self.L[idx]

def collate(batch):
    Gs,Es,Ls = zip(*batch); bg = dgl.batch(list(Gs))
    offs,acc=[],0
    for g in Gs: offs.append(acc); acc+=g.num_nodes()
    Es_shift, Ls_cat = [],[]
    for e,l,o in zip(Es,Ls,offs): Es_shift.append(e+o); Ls_cat.append(l)
    return bg, torch.cat(Es_shift), torch.cat(Ls_cat)

NUM_WORKERS = 2  
FULL = GraphDS(GRAPHS, EDGE_LISTS, LABELS)
tr, tmp = train_test_split(range(len(FULL)), test_size=0.3, random_state=42)
va, te = train_test_split(tmp, test_size=0.5, random_state=42)
mk = lambda ids,sh: DataLoader(torch.utils.data.Subset(FULL,ids), 128, sh, collate_fn=collate,
                               num_workers=NUM_WORKERS, pin_memory=True)
train_loader, val_loader, test_loader = mk(tr,True), mk(va,False), mk(te,False)

# %% Class imbalance 
pos_ratio = sum(l.sum() for l in LABELS)/sum(len(l) for l in LABELS)
pos_w = torch.tensor([(1-pos_ratio)/pos_ratio], device=device)
print(f"Global pos ratio ≈ {pos_ratio:.3f}  -> pos_weight = {pos_w.item():.2f}")

# %% Model 
from dgl.nn import NNConv
class GNN(nn.Module):
    def __init__(self,h=128,edge_dim=3):
        super().__init__()
        self.lin, self.bn0 = nn.Linear(4,h), nn.BatchNorm1d(h)
        edge_net = nn.Sequential(nn.Linear(edge_dim,h*h), nn.ReLU(), nn.BatchNorm1d(h*h))
        self.g1, self.g2 = NNConv(h,h,edge_net,'mean'), NNConv(h,h,edge_net,'mean')
        self.bn1, self.bn2 = nn.BatchNorm1d(h), nn.BatchNorm1d(h)
        self.pred = nn.Sequential(nn.Linear(2*h,h), nn.ReLU(), nn.Dropout(0.25), nn.Linear(h,1))
    def forward(self,g,idx):
        h0 = F.relu(self.bn0(self.lin(g.ndata['feat'])))
        h1 = F.relu(self.bn1(self.g1(g,h0,g.edata['feat'])))
        h1 = h0 + F.dropout(h1,0.15,self.training) # Residual skip + Dropout
        h2 = F.relu(self.bn2(self.g2(g,h1,g.edata['feat'])))
        h2 = h1 + F.dropout(h2,0.15,self.training) # Residual skip + Dropout
        s,d = idx[:,0], idx[:,1]
        return self.pred(torch.cat([h2[s],h2[d]],1)).squeeze(-1)

model = GNN().to(device)

# %% Optimizer & sched 
loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_w)
opt = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
scaler = torch.cuda.amp.GradScaler()
sched = optim.lr_scheduler.CosineAnnealingWarmRestarts(opt,T_0=4,T_mult=2,eta_min=1e-5)

# %% Evaluation 
@torch.no_grad()
def evaluate(loader, thr=0.5):
    model.eval(); logits,labs=[],[]
    for g,e,l in loader:
        g,e,l = g.to(device), e.to(device), l.to(device)
        with torch.cuda.amp.autocast(): logits.append(model(g,e).float().cpu()); labs.append(l.cpu())
    logits = torch.cat(logits); labs = torch.cat(labs).numpy()
    probs = torch.sigmoid(logits).numpy(); preds = (probs>=thr).astype(int)
    P,R,F1 = precision_score(labs,preds,zero_division=0), recall_score(labs,preds,zero_division=0), f1_score(labs,preds,zero_division=0)
    pc,rc,_ = precision_recall_curve(labs,probs); pr_auc = auc(rc,pc); acc = (preds==labs).mean()
    return acc,P,R,F1,pr_auc,probs,labs

# %% Hard‑negative mining 
@torch.no_grad()
def mine_hard_negatives():
    model.eval()
    for gi,g in enumerate(GRAPHS):
        exist = set(map(tuple,EDGE_LISTS[gi].cpu().tolist()))
        cand = [(i,j) for i in range(g.num_nodes()) for j in range(i+1,g.num_nodes()) if (i,j) not in exist]
        if not cand: continue
        batch = torch.tensor(cand, dtype=torch.long, device=device)
        with torch.cuda.amp.autocast(): scr = torch.sigmoid(model(g.to(device),batch)).cpu().numpy()
        topk = int(0.5*LABELS[gi].sum()); hard = batch[np.argsort(scr)[-topk:]].cpu()
        EDGE_LISTS[gi] = torch.cat([EDGE_LISTS[gi], hard])
        LABELS[gi] = torch.cat([LABELS[gi], torch.zeros(len(hard))])
    print('[INFO] Hard negs added')

# %% Training loop 
BEST_AUC, STATE = -math.inf, None
PATIENCE = 6               # maks. liczba epok bez poprawy
no_improve = 0
EPOCHS = 80
for ep in range(1, EPOCHS + 1):
    model.train(); tot_loss = 0.0
    for g, e, l in train_loader:
        g, e, l = g.to(device), e.to(device), l.to(device)
        opt.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast():
            loss = loss_fn(model(g, e), l)
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); tot_loss += loss.item()
    sched.step(ep)

    # walidacja 
    acc, P, R, F1, auc_pr, vprob, vlab = evaluate(val_loader)
    print(f"E{ep:02d} loss={tot_loss / len(train_loader):.4f}  val PR-AUC={auc_pr:.3f}  P={P:.3f} R={R:.3f} F1={F1:.3f}")

    if auc_pr > BEST_AUC:
        BEST_AUC, STATE = auc_pr, {"state": model.state_dict(), "vprob": vprob, "vlab": vlab}
        no_improve = 0
    else:
        no_improve += 1

    if no_improve >= PATIENCE:
        print(f"[EARLY STOP] brak poprawy przez {PATIENCE} epok. Zatrzymuję trening.")
        break

    mine_hard_negatives()

# %% Threshold & test 
model.load_state_dict(STATE['state'])
prec, rec, thr = precision_recall_curve(STATE['vlab'], STATE['vprob'])
idx = np.nanargmax(2 * prec * rec / (prec + rec + 1e-9)); best_thr = thr[idx]
print(f"Best threshold = {best_thr:.4f}  (P={prec[idx]:.3f}, R={rec[idx]:.3f})")

torch.save({
    'state': STATE['state'],
    'threshold': best_thr,
    'val_pr_auc': float(BEST_AUC),
}, 'model.pth')

acc, P, R, F1, auc_pr, *_ = evaluate(test_loader, best_thr)
print("========== TEST ==========")
print(f"PR-AUC={auc_pr:.3f}  P={P:.3f}  R={R:.3f}  F1={F1:.3f}  acc={acc:.3f}")
print("==========================")
model.load_state_dict(STATE['state'])
prec, rec, thr = precision_recall_curve(STATE['vlab'], STATE['vprob'])
idx = np.nanargmax(2*prec*rec/(prec+rec+1e-9)); best_thr = thr[idx]
print(f"\nBest threshold={best_thr:.4f} (P={prec[idx]:.3f}, R={rec[idx]:.3f})")

torch.save({'state':STATE['state'], 'threshold':best_thr, 'val_pr_auc':float(BEST_AUC)}, 'simple-model.pth')
acc,P,R,F1,auc_pr,*_ = evaluate(test_loader,best_thr)
print("\n========== TEST ==========")
print(f"PR-AUC={auc_pr:.3f} P={P:.3f} R={R:.3f} F1={F1:.3f} acc={acc:.3f}")
print("==========================")
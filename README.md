
# GraphaFold

**GraphaFold** is a deep learning pipeline for predicting non-canonical base pairings in RNA using graph neural networks. It also provides evaluation metrics and visualizations when ground truth data is available.

---

## Folder Structure

```
examples/
├── amt/   # (optional) Ground truth interaction matrices
├── cmt/   # Required. Canonical structure matrices
└── idx/   # Required. Node index mappings (with nucleotide info)
```

---

## Quickstart

### Build and run using Docker Compose

```bash
docker compose up --build
```

---

## Input Requirements

### `cmt/` (Canonical Matrix)
- CSV square matrix (`.cmt`) with 1s marking canonical edges.

### `idx/` (Index Mapping)
- `.idx` file with lines like `1,A.34`, providing node index and nucleotide.

### `amt/` (Annotated Matrix, optional)
- `.amt` file with labeled edges:
  - `1` = canonical,
  - `>1` = non-canonical,
  - `-1` = neighborhood only (no edge).

> If `amt/` is not provided, GraphaFold will **still run predictions**, but no evaluation or visual comparison will be done.

---

## Output

When `amt/` is provided:

- `/results/results.csv` — evaluation metrics (Precision, Recall, F1, INF)
- `/results/helix_outputs/*.helix` — helix format for visualization
- `/results/visualizations/*.png` — image comparison of predicted vs. actual structure (via R)

When `amt/` is not provided:

- Console output only: predicted non-canonical edges per file.

---

## Model

GraphaFold uses a GNN model (`model_v4.pth`) with the following architecture:

- **Node features:** one-hot encoding of nucleotide (A, C, G, U)
- **Edge features:** binary indicators for canonical/non-canonical
- **GNN layers:** 2x `NNConv` (mean aggregator)
- **Prediction:** binary classification of edges (non-canonical vs. not)

---

## Dependencies

Installed via Docker:

- Python 3, PyTorch, DGL
- R, `Biostrings`, `R4RNA` (from source)
- Other system libraries for R/C++ compilation

---


## License

CC-BY 4.0

---

##  Author

Built by Paulina Hladki. Contributions welcome!

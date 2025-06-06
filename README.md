
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
### Local installation
Run:
```
pip install .
```
Then you need to install the following dependencies manually:
```
pip install dgl -f https://data.dgl.ai/wheels/torch-2.3/repo.html
```
---

## Run prediction

### Input format
GraphaFold requires the RNA sequence and canonical secondary structure in dotbracket notation. This format is then parsed into a graph structure.
The input file should be in the following format:
```
>sample_1
GAGA;UAGC
(..(;)..)
```

### Run command
Then `GraphaFold` can be run with the following command:
```
graphafold -i examples/example.dot -o results/
```
The output will be saved in the `results/` in the `*.amt` format.

## Input Requirements for training and evaluation

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

## Training from scratch

### Data preparation
First, run the preprocessing script to convert your input files into the required format:
```
scripts/preprocess.py --input <path to amt and idx dir> --output <output dir>
```
### Run training
```
graphafold_train --config config/config.yml
```

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

MIT — feel free to use, modify, and share.

---

##  Author

Built by Paulina Hładki. Contributions welcome!

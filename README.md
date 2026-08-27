# GraphaFold

GraphaFold is a sequence-conditioned graph neural network for predicting
**noncanonical RNA interactions**. It combines global nucleotide context from
RiNALMo with relational message passing over the RNA backbone and a trusted
canonical-pair scaffold.

The model solves a binary contact-detection task: given an RNA sequence and
its canonical scaffold, it assigns a probability of noncanonical interaction
to candidate nucleotide pairs. Ground-truth noncanonical edges are never part
of the input graph.

## Highlights

- graph-first architecture with edge-type-aware message passing;
- frozen RiNALMo-Mega sequence embeddings in `GraphaFold` or embeddings
  learned from scratch in `GraphaFold-ablation`;
- symmetric pair scoring: `(i, j)` and `(j, i)` have identical semantics;
- complete sequence/descriptor context before candidate-pair selection;
- strict PDB-level split validation and group-balanced training;
- dynamic hard-negative sampling for the highly imbalanced contact task;
- fixed validation-selected threshold during external evaluation;
- separate `predict`, `evaluate`, `audit`, `train` and `train-ablation` commands;
- native PyTorch graph representation without DGL or PyTorch Geometric.

## Task definition

GraphaFold uses the following binary convention:

- observed `AU`/`UA`, `CG`/`GC` and `GU`/`UG` interactions are canonical;
- every observed interaction formed by another nucleotide combination is
  noncanonical;
- `0` denotes no interaction;
- `-1` denotes backbone adjacency.

This normalization is applied in memory. Source AMT and CMT files are never
modified. It is specific to the binary GraphaFold task: models that predict
Leontis--Westhof geometry classes should retain geometry-labelled AU, GC and GU
contacts as class targets.

## Installation

GraphaFold requires Python 3.10 or newer. GPU inference is strongly recommended
for the RiNALMo model, and training requires CUDA.

### Conda environment

```bash
git clone https://github.com/hlvdkv/GraphaFold.git
cd GraphaFold
conda env create -f environment.yml
conda activate graphafold
PYTHONNOUSERSITE=1 graphafold --help
```

### Existing CUDA environment

Install a PyTorch build compatible with the CUDA driver first, then install the
project and the optional RiNALMo stack:

```bash
python -m pip install -e ".[rinalmo]"
```

The tested RiNALMo environment uses `torch==2.5.1`,
`multimolecule==0.2.0`, `transformers==5.9.0` and `numpy==1.26.4`.
On a shared cluster, set `PYTHONNOUSERSITE=1` to prevent incompatible packages
from `~/.local` from shadowing the active environment.

Before a long run, verify the dependency versions and pretrained weights:

```bash
PYTHONNOUSERSITE=1 graphafold check-rinalmo
```

## Models and training data

The pretrained `GraphaFold` and `GraphaFold-ablation` checkpoints, together
with the training set, are available from Zenodo:

**[doi:10.5281/zenodo.22125162](https://doi.org/10.5281/zenodo.22125162)**

After downloading, place model files in `checkpoints/` and extract the dataset
as `TrainingSet/` or pass its location explicitly with `--data-root`.

## Quick start

### Predict new interactions

Prediction needs an `idx/` sequence directory and a matching `cmt/` canonical
scaffold directory. It does not require AMT labels.

```text
my_input/
├── idx/
│   └── molecule.idx
└── cmt/
    └── molecule.cmt
```

```bash
graphafold predict \
  --checkpoint checkpoints/GraphaFold.pt \
  --input-dir my_input \
  --output-dir outputs/my_prediction \
  --device cuda \
  --amp bf16
```

The checkpoint's validation-selected threshold and candidate-window radius are
used automatically unless explicitly overridden.

### Evaluate labelled molecules

Evaluation expects matching `amt/`, `cmt/` and `idx/` directories:

```bash
graphafold evaluate \
  --checkpoint checkpoints/GraphaFold.pt \
  --benchmark-dir path/to/benchmark \
  --output-dir outputs/benchmark \
  --device cuda \
  --amp bf16
```

Run the format, leakage and candidate-coverage audit without loading a model:

```bash
graphafold evaluate \
  --benchmark-dir path/to/benchmark \
  --output-dir outputs/benchmark-audit \
  --training-data-root path/to/TrainingSet \
  --audit-only
```

## Training

The split manifest must contain the columns `id,set`, with `set` equal to
`train`, `val` or `test`. All descriptors from one PDB ID must belong to the
same split.

```bash
graphafold audit \
  --data-root TrainingSet \
  --split-csv split.csv \
  --candidate-window 15 \
  --output label-audit.json

graphafold train \
  --data-root TrainingSet \
  --split-csv split.csv \
  --output-dir runs/GraphaFold \
  --batch-size 12 \
  --gradient-accumulation 2 \
  --workers 2 \
  --candidate-window 15 \
  --negative-ratio 128 \
  --amp bf16
```

Training intentionally refuses to start without CUDA. The command above is
intended for an A100-class node, not a login node or laptop.

In `GraphaFold-ablation`, the GNN and pair head remain unchanged but RiNALMo is
replaced with token, position and Transformer embeddings learned from scratch:

```bash
graphafold train-ablation \
  --data-root TrainingSet \
  --split-csv split.csv \
  --output-dir runs/GraphaFold-ablation \
  --batch-size 12 \
  --gradient-accumulation 2 \
  --workers 2 \
  --amp bf16
```

## Outputs

Prediction and evaluation create:

- `summary.json` -- run configuration and aggregate results;
- `per_molecule.jsonl` -- one record per processed molecule;
- `skipped.json` -- incomplete or invalid inputs and inference failures;
- `pairwise_predictions/*.csv` -- every scored candidate pair;
- `noncanonical_predictions/*.csv` -- pairs above the fixed threshold.

Evaluation additionally writes `data_audit.json` and reports both candidate-
conditional metrics and end-to-end pipeline metrics. True contacts outside the
candidate window count as false negatives in `pipeline_recall` and
`pipeline_f1`.

Training writes `config.json`, `history.jsonl`, `best.pt` and `metrics.json`.
Each history row contains paired training/validation F1 and loss values.

## Repository layout

```text
graphafold/          installable Python package
tests/               CPU unit tests
docs/                method, data-format and training documentation
checkpoints/         local model weights (not tracked by Git)
train.py             legacy-compatible training wrapper
train_ablation.py    legacy-compatible ablation wrapper
evaluate.py          legacy-compatible evaluation wrapper
audit_dataset.py     legacy-compatible audit wrapper
```

Large training data, benchmark data, checkpoints, run directories and generated
predictions are intentionally excluded from Git. Model checkpoints and the
training set are archived on
[Zenodo](https://doi.org/10.5281/zenodo.22125162). See
[`checkpoints/README.md`](checkpoints/README.md) for the expected local layout.

## Documentation

- [Methodology and leakage controls](docs/methodology.md)
- [Input and output formats](docs/data-format.md)
- [Training and reproducibility](docs/training.md)
- [Command and script guide](docs/scripts.md)

## Status and scope

GraphaFold is research software. Its predictions should be interpreted as
model-derived hypotheses, not experimental evidence. Performance depends on
the availability and quality of the canonical scaffold, and the candidate
window imposes an explicit upper bound on end-to-end recall.

## License

This project is released under the [MIT License](LICENSE).

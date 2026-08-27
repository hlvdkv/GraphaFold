# Training and reproducibility

## Hardware

Training is designed for a CUDA GPU and refuses to start when CUDA is not
available. The default RiNALMo configuration targets an A100-class device.
Data audits and unit tests are CPU-only and safe to run on a login node.

## Environment

The fully tested versions are recorded in `environment.yml` and
`requirements-rinalmo.txt`. On shared systems, isolate the environment from
user-site packages:

```bash
export PYTHONNOUSERSITE=1
```

If Conda dependency solving is slow, create a minimal Python 3.10 environment,
install the correct CUDA-enabled PyTorch wheel, and run:

```bash
python -m pip install -e ".[rinalmo]"
```

Do not mix NumPy/SciPy/scikit-learn binaries from Conda with incompatible
copies in `~/.local`.

## Pre-flight checks

Audit a small subset first:

```bash
graphafold audit \
  --data-root TrainingSet \
  --split-csv split.csv \
  --max-samples 100 \
  --output quick-audit.json
```

Then verify RiNALMo loading:

```bash
graphafold check-rinalmo
```

The second command downloads/loads the selected pretrained model and may take
several minutes on first use.

## Default optimization

The default training configuration uses:

- eight relational GNN blocks;
- hidden dimension `384`;
- asymmetric focal loss;
- AdamW with cosine decay and warm-up;
- BF16 mixed precision;
- gradient clipping and accumulation;
- at most 128 dynamically sampled negatives per positive;
- early stopping based on validation average precision.

Use `graphafold train --help` for the complete configuration surface. Every
resolved option is written to `config.json` in the run directory.

## Curves and metrics

`history.jsonl` records one row per epoch with:

- `train_loss` and `val_loss`;
- `train_f1` and `val_f1` at the validation-selected threshold;
- validation average precision and threshold;
- current learning rate.

Training F1 is diagnostic: it is computed on the dynamically sampled training
pairs. Validation F1 and loss are computed on the complete validation candidate
distribution, which is much more imbalanced. Their absolute values should not
be interpreted as if they came from identical sampling distributions.

## Reproducibility artefacts

Keep the following together for each reported experiment:

- source-code commit hash;
- `config.json`;
- `history.jsonl`;
- `metrics.json`;
- `best.pt`;
- split manifest and data-audit JSON;
- exact environment export and GPU model.

The external-test threshold must come from the checkpoint. Do not choose a new
threshold on the benchmark.


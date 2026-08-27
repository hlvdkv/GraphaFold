# Command and script guide

The installed `graphafold` command is the supported public interface.

| Command | Purpose | Requires GPU |
|---|---|---|
| `graphafold predict` | Predict from IDX + CMT without ground truth | recommended for RiNALMo |
| `graphafold evaluate` | Evaluate complete IDX/CMT/AMT triples | recommended for RiNALMo |
| `graphafold evaluate --audit-only` | Validate benchmark format and overlap | no |
| `graphafold audit` | Audit training labels, imbalance and candidate coverage | no |
| `graphafold train` | Train `GraphaFold` with RiNALMo | yes |
| `graphafold train-ablation` | Train `GraphaFold-ablation` without RiNALMo | yes |
| `graphafold check-rinalmo` | Verify dependencies and pretrained weights | no, but memory intensive |

The repository-root scripts `train.py`, `train_ablation.py`, `evaluate.py`,
`audit_dataset.py` and `check_rinalmo.py` remain available for compatibility
with existing cluster job files.

The remaining root-level `analyze_*`, `build_*` and comparison scripts are
research utilities used to reproduce thesis analyses. They are not required by
the installed package or by ordinary prediction.

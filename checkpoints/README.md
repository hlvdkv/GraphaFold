# Checkpoints

Model weights are intentionally not stored in Git because the RiNALMo-backed
checkpoint exceeds GitHub's regular file-size limit.

The `GraphaFold` and `GraphaFold-ablation` checkpoints and the complete training
set can be downloaded from Zenodo:

**[doi:10.5281/zenodo.22125162](https://doi.org/10.5281/zenodo.22125162)**

Place downloaded checkpoints in this directory, for example:

```text
checkpoints/
├── GraphaFold.pt
└── GraphaFold-ablation.pt
```

The checkpoint selected on validation stores the architecture configuration,
classification threshold and candidate-window radius required for inference.
Keep downloaded model weights outside Git and reference their local paths with
`--checkpoint`.

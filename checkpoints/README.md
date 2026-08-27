# Checkpoints

Model weights are intentionally not stored in Git because the RiNALMo-backed
checkpoint exceeds GitHub's regular file-size limit.

Place downloaded checkpoints in this directory, for example:

```text
checkpoints/
├── GraphaFold.pt
└── GraphaFold-ablation.pt
```

The checkpoint selected on validation stores the architecture configuration,
classification threshold and candidate-window radius required for inference.
Publish large checkpoints as GitHub Release assets or through a dedicated
model repository rather than committing them directly.

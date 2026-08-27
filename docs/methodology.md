# Methodology

## Model input

Each RNA molecule or training descriptor is represented as a typed graph:

- nodes are nucleotides with sequence positions;
- type `0` edges represent the RNA backbone;
- type `1` edges represent trusted canonical contacts from CMT;
- target noncanonical contacts are never added to the input graph.

The canonical input is filtered using nucleotide identity. A CMT entry marked
as canonical is used only for AU/UA, CG/GC or GU/UG. This prevents an erroneous
or target-derived CMT entry from leaking a noncanonical label through graph
topology.

## Architecture

The default model has three principal stages:

1. **Sequence encoder.** Frozen RiNALMo-Mega embeddings supply global sequence
   context and are projected into the GraphaFold hidden dimension.
2. **Relational GNN.** Eight gated message-passing blocks update nucleotide
   representations using backbone and canonical edge types.
3. **Symmetric pair head.** Candidate pairs are represented using the sum,
   absolute difference and element-wise product of endpoint states, together
   with base-combination and sequence-distance embeddings.

`GraphaFold-ablation` replaces RiNALMo with nucleotide and positional
embeddings plus a small Transformer. The graph encoder, candidate generator
and pair head remain the same.

## Candidate generation

The full sequence or descriptor is encoded before candidate selection. By
default, candidate pairs are formed from the Cartesian product of the
`+/-15`-nucleotide neighbourhoods around the two endpoints of every canonical
pair. Backbone pairs are removed.

The window limits quadratic pair scoring but also creates a hard recall ceiling.
For that reason, every audit and evaluation reports candidate recall separately
from classification recall. Setting the radius to `0` enables all-pairs scoring
for diagnostic experiments; it is not the default methodology.

## Label normalization

The binary target is derived from AMT and nucleotide identities:

| AMT state | Base combination | GraphaFold target |
|---|---|---|
| `-1` | any | backbone |
| `0` | any | no interaction |
| `>0` | AU/UA, CG/GC, GU/UG | canonical |
| `>0` | any other combination | noncanonical |

This convention deliberately differs from geometry-class prediction. An AU,
GC or GU contact with a noncanonical Leontis--Westhof geometry remains a valid
geometry target in a multiclass model, but is canonical for GraphaFold's binary
base-combination task.

## Imbalance handling

The positive class is rare. Training therefore rotates a bounded set of
negatives between epochs and favours difficult negatives with noncanonical base
compositions. Validation and external evaluation do not subsample candidates:
every pair admitted by the candidate generator is scored.

Training examples are balanced at the PDB-group level so that structures with
many descriptors do not dominate an epoch.

## Threshold and checkpoint selection

The classification threshold is selected on the validation set and stored in
the checkpoint. It does not participate in the loss, backward pass, optimizer
or learning-rate schedule. External prediction and evaluation reuse this fixed
threshold by default and never optimize a new threshold on test data.

The best checkpoint is selected by validation average precision, which is more
informative than accuracy for a rare-positive ranking problem.

## Leakage controls

- PDB groups cannot occur in more than one manifest split.
- Noncanonical target edges are absent from the input graph.
- CMT edges inconsistent with the canonical base-combination rule are rejected.
- External benchmark audits can check PDB-ID and exact full-sequence overlap
  against the training directory.
- Validation scores the complete candidate distribution rather than the
  negative sample used during optimization.

## Reported metrics

Candidate metrics describe classification among pairs selected by the window.
Pipeline metrics additionally count true noncanonical contacts outside the
window as false negatives:

```text
pipeline recall = correctly predicted NC contacts / all true NC contacts
```

Both micro aggregates and macro per-molecule values are written during external
evaluation. For comparisons across molecules of very different lengths, macro
F1 should be reported explicitly alongside the candidate recall ceiling.

# Data formats

GraphaFold uses comma-separated IDX, CMT and AMT files. Files belonging to one
molecule must share the same stem.

## Prediction input

Ground-truth-free prediction requires:

```text
input_directory/
├── idx/
│   ├── molecule_1.idx
│   └── molecule_2.idx
└── cmt/
    ├── molecule_1.cmt
    └── molecule_2.cmt
```

## Evaluation input

Evaluation additionally requires AMT ground truth:

```text
benchmark_directory/
├── idx/
├── cmt/
└── amt/
```

## IDX

IDX stores one nucleotide per line using one-based, consecutive indices:

```text
1,A.A1
2,A.U2
3,A.G3
4,A.C4
```

The first field is the matrix position. The second field is a residue identifier;
the nucleotide is read from the part following the first dot. `T` is normalized
to `U`; unsupported symbols become `N`.

## CMT

CMT is an `N x N`, symmetric, integer CSV matrix aligned with IDX:

- `-1` -- backbone adjacency;
- `0` -- no supplied canonical edge;
- `1` -- canonical contact.

The diagonal must contain zeros. For graph construction, a value of `1` is
accepted only when the two bases form AU/UA, CG/GC or GU/UG.

## AMT

AMT is required for training and evaluation, but not prediction. It is an
`N x N`, symmetric, integer CSV matrix aligned with IDX:

- `-1` -- backbone adjacency;
- `0` -- no interaction;
- `1..13` -- observed interaction in the source annotation.

GraphaFold converts these values to its binary target in memory according to
the base-combination rule described in
[`methodology.md`](methodology.md#label-normalization).

## Split manifest

Training uses a CSV file with a header and one row per descriptor:

```csv
id,set
4V88_1_A6_A_1490_C,train
example_descriptor,val
```

The PDB group is the part before the first underscore. The loader rejects a
manifest if one PDB group occurs in multiple splits.

## Prediction CSV

`pairwise_predictions/<id>.csv` contains every scored candidate. The compact
`noncanonical_predictions/<id>.csv` contains only rows whose probability is at
least the fixed threshold. Prediction columns include:

- one-based nucleotide indices and original residue identifiers;
- nucleotide identities and sequence distance;
- whether the pair was a known canonical input;
- noncanonical probability and binary prediction.

Evaluation adds the raw AMT label and true binary target to these columns.


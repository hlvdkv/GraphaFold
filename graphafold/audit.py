"""CPU-only audit utilities for GraphaFold training datasets."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .data import _read_square_matrix, read_idx, read_manifest, window_candidates
from .labels import (
    InteractionKind,
    base_pair_code,
    classify_interaction,
    is_canonical_base_pair,
)


def audit_training_data(
    data_root: str | Path,
    split_csv: str | Path,
    candidate_window: int = 15,
    max_samples: int = 0,
) -> dict[str, Any]:
    """Audit labels, class balance and candidate coverage without modifying data."""

    records = read_manifest(data_root, split_csv)
    if max_samples:
        records = records[:max_samples]
    source_labels: Counter[str] = Counter()
    normalized: Counter[str] = Counter()
    cmt_canonical: Counter[str] = Counter()
    reclassified: Counter[str] = Counter()
    by_split: dict[str, Counter[str]] = defaultdict(Counter)
    candidate_pool_by_split: dict[str, Counter[str]] = defaultdict(Counter)
    candidate_positive_count = 0
    total_positive_count = 0

    for index, record in enumerate(records, start=1):
        bases = read_idx(record.idx_path)
        amt = _read_square_matrix(record.amt_path)
        cmt = _read_square_matrix(record.cmt_path)
        size = min(len(bases), amt.shape[0], cmt.shape[0])
        sample_positives: set[tuple[int, int]] = set()
        sample_canonical: set[tuple[int, int]] = set()
        sample_backbone: set[tuple[int, int]] = set()
        for i, j in zip(*np.triu_indices(size, k=1)):
            raw = int(amt[i, j])
            pair = base_pair_code(bases[i], bases[j])
            if raw == -1:
                sample_backbone.add((int(i), int(j)))
            if cmt[i, j] == 1:
                cmt_canonical[pair] += 1
                if is_canonical_base_pair(bases[i], bases[j]):
                    sample_canonical.add((int(i), int(j)))
            if raw <= 0:
                continue
            kind = classify_interaction(raw, bases[i], bases[j])
            source_labels[f"{pair}:label={raw}"] += 1
            normalized[f"{pair}:{kind.name.lower()}"] += 1
            by_split[record.split][kind.name.lower()] += 1
            if kind == InteractionKind.NONCANONICAL:
                sample_positives.add((int(i), int(j)))
            if raw > 1 and kind == InteractionKind.CANONICAL:
                reclassified[pair] += 1
        candidates = window_candidates(size, sample_canonical, candidate_window) - sample_backbone
        candidate_positives = len(sample_positives & candidates)
        candidate_negatives = len(candidates - sample_positives)
        candidate_positive_count += candidate_positives
        total_positive_count += len(sample_positives)
        candidate_pool_by_split[record.split]["records"] += 1
        candidate_pool_by_split[record.split]["positive"] += candidate_positives
        candidate_pool_by_split[record.split]["negative"] += candidate_negatives
        if index % 5000 == 0:
            print(f"audited {index}/{len(records)}", flush=True)

    candidate_pool_report: dict[str, dict[str, Any]] = {}
    for split, counts in candidate_pool_by_split.items():
        positives = counts["positive"]
        negatives = counts["negative"]
        candidate_pool_report[split] = {
            **dict(counts),
            "negative_per_positive": negatives / max(1, positives),
            "positive_prevalence": positives / max(1, positives + negatives),
        }

    return {
        "samples": len(records),
        "canonical_rule": ["AU", "CG", "GU"],
        "cmt_pairs_marked_canonical": dict(cmt_canonical),
        "source_positive_labels": dict(source_labels),
        "normalized_interactions": dict(normalized),
        "raw_label_gt1_reclassified_as_canonical": dict(reclassified),
        "normalized_by_split": {split: dict(counts) for split, counts in by_split.items()},
        "candidate_pool_by_split": candidate_pool_report,
        "candidate_window": candidate_window,
        "candidate_positives": candidate_positive_count,
        "total_noncanonical_positives": total_positive_count,
        "candidate_recall": candidate_positive_count / max(1, total_positive_count),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit source and normalized GraphaFold labels")
    parser.add_argument("--data-root", default="TrainingSet")
    parser.add_argument("--split-csv", default="split.csv")
    parser.add_argument("--max-samples", type=int, default=0, help="0 scans the complete manifest")
    parser.add_argument("--candidate-window", type=int, default=15)
    parser.add_argument("--output", default="label-audit.json")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = audit_training_data(
        data_root=args.data_root,
        split_csv=args.split_csv,
        candidate_window=args.candidate_window,
        max_samples=args.max_samples,
    )
    Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


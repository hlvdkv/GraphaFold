#!/usr/bin/env python3
"""Audit the shared descriptor corpus for SDT and GraphaFold thesis reporting.

This is a read-only, CPU analysis. It does not import either model or start
training. The current raw TrainingSet/split.csv are treated as the reconstructed
source corpus; the prepared SDT `pairwise_dataset` is not part of the archive.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

# The analysis is also run on headless cluster/login nodes and from macOS
# terminal sessions.  Force a non-interactive backend before importing pyplot.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


CANONICAL_COMPOSITIONS = {"AU", "CG", "GU"}
PAIR_TYPES = ("AA", "AC", "AG", "AU", "CC", "CG", "CU", "GG", "GU", "UU", "N/other")
CLASS_NAMES = {
    1: "canonical/raw",
    2: "tWW", 3: "tHH", 4: "tSS", 5: "tWS/tSW", 6: "tWH/tHW",
    7: "tHS/tSH", 8: "cWW", 9: "cHH", 10: "cSS", 11: "cWS/cSW",
    12: "cWH/cHW", 13: "cHS/cSH",
}
BLUE = "#4E638F"
LIGHT_PURPLE = "#F3E5F5"
PURPLE = "#C693C7"
ORANGE = "#F0B35A"
LIGHT_BLUE = "#9BB9D4"
COLORS = {"train": BLUE, "val": PURPLE, "SDT": PURPLE, "GraphaFold": BLUE}


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        rows = [{"id": row["id"].strip(), "split": row["set"].strip().lower()} for row in csv.DictReader(handle)]
    return [row for row in rows if row["id"] and row["split"] in {"train", "val"}]


def read_idx(path: Path) -> tuple[list[str], list[str]]:
    """Mirror graphafold.data.read_idx, including possible index gaps."""
    indexed: dict[int, tuple[str, str]] = {}
    for raw in path.read_text().splitlines():
        if not raw.strip():
            continue
        index_text, code = raw.split(",", 1)
        index = int(index_text) - 1
        chain, token = code.split(".", 1) if "." in code else ("", code)
        base = token[0].upper().replace("T", "U")
        if base not in "ACGU":
            base = "N"
        indexed[index] = (base, chain)
    if not indexed:
        raise ValueError(f"Empty IDX: {path}")
    bases = ["N"] * (max(indexed) + 1)
    chains = [""] * len(bases)
    for index, (base, chain) in indexed.items():
        bases[index] = base
        chains[index] = chain
    return bases, chains


def pair_code(a: str, b: str) -> str:
    return "".join(sorted((a, b))) if a in "ACGU" and b in "ACGU" else "N/other"


def pad_square(matrix: np.ndarray, size: int) -> np.ndarray:
    if matrix.shape[0] == size:
        return matrix
    if matrix.shape[0] > size:
        return matrix[:size, :size]
    return np.pad(matrix, ((0, size - matrix.shape[0]), (0, size - matrix.shape[1])))


def graph_counts(
    bases: list[str], amt: np.ndarray, cmt: np.ndarray, size: int,
    candidate_window: int = 15, include_candidates: bool = False,
) -> dict[str, int]:
    """Reproduce GraphaFold target and candidate construction at a chosen size."""
    padded_bases = bases + ["N"] * (size - len(bases))
    padded_amt = pad_square(amt, size)
    padded_cmt = pad_square(cmt, size)
    backbone: set[tuple[int, int]] = set()
    canonical_input: set[tuple[int, int]] = set()
    positives: set[tuple[int, int]] = set()
    canonical_targets = 0
    for i, j in zip(*np.triu_indices(size, k=1)):
        raw = int(padded_amt[i, j])
        composition = pair_code(padded_bases[i], padded_bases[j])
        if raw == -1:
            backbone.add((int(i), int(j)))
        if padded_cmt[i, j] == 1 and composition in CANONICAL_COMPOSITIONS:
            canonical_input.add((int(i), int(j)))
        if raw > 0:
            if composition in CANONICAL_COMPOSITIONS:
                canonical_targets += 1
            else:
                positives.add((int(i), int(j)))
    if not include_candidates:
        return {
            "canonical": canonical_targets, "noncanonical": len(positives),
            "candidate_positive": 0, "candidate_negative": 0,
        }
    candidates: set[tuple[int, int]] = set()
    for left, right in canonical_input:
        for i in range(max(0, left - candidate_window), min(size, left + candidate_window + 1)):
            for j in range(max(0, right - candidate_window), min(size, right + candidate_window + 1)):
                if i != j:
                    candidates.add((min(i, j), max(i, j)))
    candidates -= backbone
    return {
        "canonical": canonical_targets,
        "noncanonical": len(positives),
        "candidate_positive": len(positives & candidates),
        "candidate_negative": len(candidates - positives),
    }


def quantiles(values: list[int]) -> dict[str, float]:
    array = np.asarray(values)
    return {
        "min": int(array.min()), "q05": float(np.quantile(array, 0.05)),
        "q25": float(np.quantile(array, 0.25)), "median": float(np.median(array)),
        "mean": float(array.mean()), "q75": float(np.quantile(array, 0.75)),
        "q95": float(np.quantile(array, 0.95)), "max": int(array.max()),
    }


def format_count(value: int | float) -> str:
    return f"{int(value):,}".replace(",", " ")


def format_decimal(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}".replace(".", ",")


def savefig(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split-csv", type=Path, required=True)
    parser.add_argument("--label-audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    figures = args.output_dir / "figures"
    figures.mkdir(exist_ok=True)
    plt.rcParams.update({
        "figure.figsize": (8.2, 5.1), "font.size": 10.5, "axes.titlesize": 13,
        "axes.labelsize": 11, "axes.grid": False, "axes.edgecolor": "#444444",
        "axes.linewidth": 1.0, "axes.spines.top": True, "axes.spines.right": True,
    })

    manifest = read_manifest(args.split_csv)
    audit = json.loads(args.label_audit.read_text())
    availability = Counter()
    alignment = Counter()
    label_counts = {"train": Counter(), "val": Counter()}
    sdt_eligible_label_counts = {"train": Counter(), "val": Counter()}
    pair_class_counts = {"train": Counter(), "val": Counter()}
    pair_type_raw_nc = {"train": Counter(), "val": Counter()}
    pair_type_graph_nc = {"train": Counter(), "val": Counter()}
    reclassified = {"train": Counter(), "val": Counter()}
    promoted = {"train": Counter(), "val": Counter()}
    normalized = {"train": Counter(), "val": Counter()}
    normalized_training_semantics = {"train": Counter(), "val": Counter()}
    candidate_semantics = {
        "audit_min": {"train": Counter(), "val": Counter()},
        "training_max": {"train": Counter(), "val": Counter()},
    }
    lengths = {"train": [], "val": []}
    groups = {"train": Counter(), "val": Counter()}
    sdt_records = Counter()
    sequence_groups: dict[str, list[tuple[str, str, str, str]]] = defaultdict(list)
    sample_rows: list[dict[str, object]] = []
    tri_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    for index, row in enumerate(manifest, start=1):
        sid, split = row["id"], row["split"]
        idx_path = args.data_root / "idx" / f"{sid}.idx"
        amt_path = args.data_root / "amt" / f"{sid}.amt"
        cmt_path = args.data_root / "cmt" / f"{sid}.cmt"
        has_idx, has_amt, has_cmt = idx_path.is_file(), amt_path.is_file(), cmt_path.is_file()
        availability[(has_idx, has_amt, has_cmt)] += 1
        if not (has_idx and has_amt and has_cmt):
            continue
        bases, chains = read_idx(idx_path)
        amt = np.loadtxt(amt_path, delimiter=",", dtype=np.int16, ndmin=2)
        cmt = np.loadtxt(cmt_path, delimiter=",", dtype=np.int16, ndmin=2)
        if amt.shape[0] != amt.shape[1] or cmt.shape[0] != cmt.shape[1]:
            availability["non_square_matrix"] += 1
            continue
        idx_n, amt_n, cmt_n = len(bases), amt.shape[0], cmt.shape[0]
        n = min(idx_n, amt_n, cmt_n)  # policy used by audit_dataset.py
        training_n = max(idx_n, amt_n, cmt_n)  # policy used by load_graph()
        alignment["all_three_equal" if idx_n == amt_n == cmt_n else "dimension_mismatch"] += 1
        alignment["idx_amt_mismatch"] += int(idx_n != amt_n)
        alignment["idx_cmt_mismatch"] += int(idx_n != cmt_n)
        alignment["amt_cmt_mismatch"] += int(amt_n != cmt_n)
        alignment["nodes_ignored_by_audit_min"] += training_n - n
        sdt_eligible = idx_n == amt_n and all(base in "ACGU" for base in bases)
        if sdt_eligible:
            sdt_records[split] += 1
        if n not in tri_cache:
            tri_cache[n] = np.triu_indices(n, k=1)
        tri_i, tri_j = tri_cache[n]
        values = amt[tri_i, tri_j]
        unique, counts = np.unique(values, return_counts=True)
        label_counts[split].update({int(k): int(v) for k, v in zip(unique, counts)})
        if sdt_eligible:
            sdt_i, sdt_j = np.triu_indices(idx_n, k=1)
            sdt_values = amt[sdt_i, sdt_j]
            sdt_unique, sdt_counts = np.unique(sdt_values, return_counts=True)
            sdt_eligible_label_counts[split].update({int(k): int(v) for k, v in zip(sdt_unique, sdt_counts)})

        raw_canonical = int(np.sum(values == 1))
        raw_nc = int(np.sum(values > 1))
        backbone = int(np.sum(values == -1))
        no_pair = int(np.sum(values == 0))
        graph_canonical = 0
        graph_nc = 0
        sample_reclassified = 0
        sample_promoted = 0
        positive_positions = np.flatnonzero(values > 0)
        for pos in positive_positions:
            i, j, raw_label = int(tri_i[pos]), int(tri_j[pos]), int(values[pos])
            composition = pair_code(bases[i], bases[j])
            pair_class_counts[split][(raw_label, composition)] += 1
            if raw_label > 1:
                pair_type_raw_nc[split][composition] += 1
            if composition in CANONICAL_COMPOSITIONS:
                graph_canonical += 1
                if raw_label > 1:
                    reclassified[split][composition] += 1
                    sample_reclassified += 1
            else:
                graph_nc += 1
                pair_type_graph_nc[split][composition] += 1
                if raw_label == 1:
                    promoted[split][composition] += 1
                    sample_promoted += 1
        normalized[split]["canonical"] += graph_canonical
        normalized[split]["noncanonical"] += graph_nc
        actual = graph_counts(bases, amt, cmt, training_n)
        normalized_training_semantics[split]["canonical"] += actual["canonical"]
        normalized_training_semantics[split]["noncanonical"] += actual["noncanonical"]
        if training_n != n:
            actual = graph_counts(bases, amt, cmt, training_n, include_candidates=True)
            audited = graph_counts(bases[:n], amt[:n, :n], cmt[:n, :n], n, include_candidates=True)
            for key in ("candidate_positive", "candidate_negative"):
                candidate_semantics["audit_min"][split][key] += audited[key]
                candidate_semantics["training_max"][split][key] += actual[key]
        lengths[split].append(n)
        group = sid.split("_", 1)[0].upper()
        groups[split][group] += 1
        if sdt_eligible:
            sequence = "".join(bases)
            label_signature = hashlib.sha1(sdt_values.tobytes()).hexdigest()
            sequence_groups[sequence].append((sid, split, group, label_signature))
        sample_rows.append({
            "sample_id": sid, "split": split, "PDB_group": group,
            "idx_length": idx_n, "amt_size": amt_n, "cmt_size": cmt_n,
            "audit_length_min": n, "training_length_max": training_n,
            "SDT_builder_eligible": int(sdt_eligible),
            "chains": len({chain for chain in chains if chain}), "upper_triangle_pairs": len(values),
            "backbone_minus1": backbone, "no_pair_0": no_pair,
            "raw_canonical_label1": raw_canonical, "raw_noncanonical_label2_13": raw_nc,
            "graphafold_canonical": graph_canonical, "graphafold_noncanonical": graph_nc,
            "reclassified_AU_CG_GU_gt1": sample_reclassified,
            "promoted_other_composition_label1": sample_promoted,
        })
        if index % 5000 == 0:
            print(f"scanned {index}/{len(manifest)}", flush=True)

    complete_triples = availability[(True, True, True)]
    complete_sdt = sum(sdt_records.values())
    split_records = {split: sum(groups[split].values()) for split in ("train", "val")}
    split_groups = {split: len(groups[split]) for split in ("train", "val")}
    cross_split_groups = set(groups["train"]) & set(groups["val"])

    duplicate_sequence_groups = {seq: rows for seq, rows in sequence_groups.items() if len(rows) > 1}
    ambiguous_sequence_groups = {
        seq: rows for seq, rows in duplicate_sequence_groups.items() if len({row[3] for row in rows}) > 1
    }
    cross_split_sequences = {
        seq: rows for seq, rows in sequence_groups.items() if {row[1] for row in rows} == {"train", "val"}
    }
    cross_split_ambiguous = {
        seq: rows for seq, rows in cross_split_sequences.items() if len({row[3] for row in rows}) > 1
    }

    raw_by_split = {
        split: {
            "backbone": label_counts[split][-1], "none": label_counts[split][0],
            "canonical_label1": label_counts[split][1],
            "noncanonical_labels2_13": sum(label_counts[split][label] for label in range(2, 14)),
            "observed_total": sum(label_counts[split][label] for label in range(1, 14)),
            "upper_triangle_total": sum(label_counts[split].values()),
        }
        for split in ("train", "val")
    }
    sdt_raw_by_split = {
        split: {
            "backbone": sdt_eligible_label_counts[split][-1],
            "none": sdt_eligible_label_counts[split][0],
            "canonical_label1": sdt_eligible_label_counts[split][1],
            "noncanonical_labels2_13": sum(sdt_eligible_label_counts[split][label] for label in range(2, 14)),
        }
        for split in ("train", "val")
    }

    class_rows = []
    for split in ("train", "val"):
        for label in range(1, 14):
            class_rows.append({"split": split, "class": label, "class_name": CLASS_NAMES[label], "count": label_counts[split][label]})
    pair_rows = []
    for split in ("train", "val"):
        for composition in PAIR_TYPES:
            pair_rows.append({
                "split": split, "pair_type": composition,
                "SDT_raw_labels_2_13": pair_type_raw_nc[split][composition],
                "GraphaFold_binary_positive": pair_type_graph_nc[split][composition],
            })
    normalization_rows = []
    for split in ("train", "val"):
        normalization_rows.append({
            "split": split,
            "raw_label1": raw_by_split[split]["canonical_label1"],
            "raw_labels2_13": raw_by_split[split]["noncanonical_labels2_13"],
            "GraphaFold_canonical": normalized[split]["canonical"],
            "GraphaFold_noncanonical": normalized[split]["noncanonical"],
            "reclassified_gt1_AU_CG_GU": sum(reclassified[split].values()),
            "promoted_label1_other": sum(promoted[split].values()),
        })

    write_csv(args.output_dir / "per_descriptor_statistics.csv", sample_rows, list(sample_rows[0]))
    write_csv(args.output_dir / "class_distribution_by_split.csv", class_rows, list(class_rows[0]))
    write_csv(args.output_dir / "pair_type_definition_comparison.csv", pair_rows, list(pair_rows[0]))
    write_csv(args.output_dir / "label_normalization_by_split.csv", normalization_rows, list(normalization_rows[0]))
    group_rows = [
        {"split": split, "PDB_group": group, "descriptors": count}
        for split in ("train", "val") for group, count in sorted(groups[split].items())
    ]
    write_csv(args.output_dir / "pdb_group_sizes.csv", group_rows, list(group_rows[0]))
    duplicate_rows = []
    for sequence, rows in sorted(duplicate_sequence_groups.items(), key=lambda item: (-len(item[1]), item[0])):
        duplicate_rows.append({
            "sequence_sha1": hashlib.sha1(sequence.encode()).hexdigest(), "length": len(sequence),
            "occurrences": len(rows), "splits": ",".join(sorted({row[1] for row in rows})),
            "PDB_groups": len({row[2] for row in rows}), "distinct_label_matrices": len({row[3] for row in rows}),
            "sample_ids": ";".join(row[0] for row in rows),
        })
    write_csv(args.output_dir / "duplicate_sequence_groups.csv", duplicate_rows, list(duplicate_rows[0]) if duplicate_rows else ["sequence_sha1"])

    # Figure 1: corpus sizes.  The number of PDB groups belongs in the text/table,
    # not in this visual comparison of the train and validation subsets.
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    x = np.arange(2)
    bars = ax.bar(
        x,
        [split_records["train"], split_records["val"]],
        color=[COLORS["train"], COLORS["val"]],
        edgecolor=BLUE,
        linewidth=0.9,
    )
    ax.set_xticks(x, ["treningowy", "walidacyjny"])
    ax.set_ylabel("Liczba deskryptorów")
    for bar in bars:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            format_count(bar.get_height()),
            ha="center",
            va="bottom",
        )
    savefig(figures / "01_split_sizes.png")

    # Figure 2: descriptor lengths.
    bins = np.arange(0, max(max(lengths["train"]), max(lengths["val"])) + 3, 2)
    plt.figure(figsize=(8.5, 5.2))
    plt.hist(lengths["train"], bins=bins, density=True, alpha=0.65, label="zbiór treningowy", color=COLORS["train"])
    plt.hist(lengths["val"], bins=bins, density=True, alpha=0.55, label="zbiór walidacyjny", color=COLORS["val"])
    plt.xlabel("Długość deskryptora [nt]"); plt.ylabel("Gęstość")
    plt.legend(frameon=False)
    savefig(figures / "02_length_distribution.png")

    # Figure 3: raw SDT class distribution.
    train_counts = np.array([label_counts["train"][label] for label in range(1, 14)])
    val_counts = np.array([label_counts["val"][label] for label in range(1, 14)])
    x = np.arange(13); width = 0.42
    plt.figure(figsize=(11, 5.3))
    plt.bar(x - width/2, train_counts, width, label="zbiór treningowy", color=COLORS["train"])
    plt.bar(x + width/2, val_counts, width, label="zbiór walidacyjny", color=COLORS["val"])
    plt.yscale("log"); plt.xticks(x, [str(i) for i in range(1, 14)])
    plt.xlabel("Surowa klasa AMT"); plt.ylabel("Liczba kontaktów (skala logarytmiczna)")
    plt.legend(frameon=False)
    savefig(figures / "03_sdt_class_distribution.png")

    # Figure 4: task-definition shift.
    raw_total = [sum(row[key] for row in normalization_rows) for key in ("raw_label1", "raw_labels2_13")]
    graph_total = [sum(row[key] for row in normalization_rows) for key in ("GraphaFold_canonical", "GraphaFold_noncanonical")]
    x = np.arange(2); width = 0.36
    plt.figure(figsize=(8.4, 5.2))
    plt.bar(x - width/2, raw_total, width, label="SDT: surowe AMT", color=COLORS["SDT"])
    plt.bar(x + width/2, graph_total, width, label="GraphaFold: po normalizacji", color=COLORS["GraphaFold"])
    plt.xticks(x, ["kanoniczne / klasa 1", "niekanoniczne / dodatnie"])
    plt.ylabel("Liczba zaobserwowanych kontaktów")
    plt.legend(frameon=False)
    for patch in plt.gca().patches:
        plt.text(patch.get_x()+patch.get_width()/2, patch.get_height(), format_count(patch.get_height()), ha="center", va="bottom", fontsize=9)
    savefig(figures / "04_label_definition_shift.png")

    # Figure 5: reclassification composition.
    rec = [sum(reclassified[s][p] for s in ("train", "val")) for p in ("AU", "CG", "GU")]
    plt.figure(figsize=(7.8, 4.9))
    bars = plt.bar(
        ("AU", "CG", "GU"),
        rec,
        color=PURPLE,
        edgecolor=BLUE,
        linewidth=1.0,
    )
    plt.ylabel("Liczba etykiet >1 przeklasyfikowanych jako kanoniczne")
    for bar, value in zip(bars, rec):
        plt.text(bar.get_x()+bar.get_width()/2, value, format_count(value), ha="center", va="bottom")
    savefig(figures / "05_reclassified_au_cg_gu.png")

    # Figure 6: class x composition heat map.
    heat = np.zeros((12, len(PAIR_TYPES)), dtype=int)
    for class_index, label in enumerate(range(2, 14)):
        for pair_index, composition in enumerate(PAIR_TYPES):
            heat[class_index, pair_index] = sum(pair_class_counts[s][(label, composition)] for s in ("train", "val"))
    fig, ax = plt.subplots(figsize=(11, 6.5))
    f1_cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "sdt_liczebnosc", ["#F7F7F7", LIGHT_PURPLE, PURPLE, BLUE]
    )
    image = ax.imshow(np.log10(heat + 1), aspect="auto", cmap=f1_cmap)
    ax.set_xticks(range(len(PAIR_TYPES)), ["N/inne" if p == "N/other" else p for p in PAIR_TYPES])
    ax.set_yticks(range(12), [f"{label}: {CLASS_NAMES[label]}" for label in range(2, 14)])
    ax.set_xlabel("Nieuporządkowany skład pary zasad"); ax.set_ylabel("Klasa Leontisa–Westhofa")
    for i in range(12):
        for j in range(len(PAIR_TYPES)):
            if heat[i, j]:
                label = f"{format_decimal(heat[i,j] / 1000, 1)} tys." if heat[i,j] >= 1000 else str(heat[i,j])
                ax.text(j, i, label, ha="center", va="center", fontsize=7, color="white" if np.log10(heat[i,j]+1)>2.5 else "black")
    fig.colorbar(image, ax=ax, label="log10(liczebność + 1)")
    savefig(figures / "06_class_composition_heatmap.png")

    # Figure 7: pair types under both task definitions.
    sdt_pair = [sum(pair_type_raw_nc[s][p] for s in ("train", "val")) for p in PAIR_TYPES]
    graph_pair = [sum(pair_type_graph_nc[s][p] for s in ("train", "val")) for p in PAIR_TYPES]
    x = np.arange(len(PAIR_TYPES)); width = 0.40
    plt.figure(figsize=(10, 5.3))
    plt.bar(x-width/2, sdt_pair, width, label="SDT: surowe klasy 2–13", color=COLORS["SDT"])
    plt.bar(x+width/2, graph_pair, width, label="GraphaFold: dodatnie binarne", color=COLORS["GraphaFold"])
    plt.yscale("symlog", linthresh=1); plt.xticks(x, ["N/inne" if p == "N/other" else p for p in PAIR_TYPES])
    plt.xlabel("Nieuporządkowany skład pary zasad"); plt.ylabel("Liczba kontaktów docelowych (skala symlog)")
    plt.legend(frameon=False)
    savefig(figures / "07_pair_type_task_comparison.png")

    # Figure 8: candidate imbalance from the authoritative GraphaFold audit.
    pool = audit["candidate_pool_by_split"]
    x = np.arange(2); width = 0.38
    positives = [pool[s]["positive"] for s in ("train", "val")]
    negatives = [pool[s]["negative"] for s in ("train", "val")]
    plt.figure(figsize=(8.3, 5.1))
    plt.bar(x-width/2, positives, width, label="dodatnie", color=PURPLE)
    plt.bar(x+width/2, negatives, width, label="ujemne", color=BLUE)
    plt.yscale("log"); plt.xticks(x, ["treningowy", "walidacyjny"])
    plt.ylabel("Liczba par kandydackich (skala logarytmiczna)")
    plt.legend(frameon=False)
    savefig(figures / "08_graphafold_candidate_imbalance.png")

    # Figure 9: PDB descriptor imbalance and epoch cap.
    ranked = sorted((count for split in ("train", "val") for count in groups[split].values()), reverse=True)
    plt.figure(figsize=(8.8, 5.0))
    plt.plot(np.arange(1, len(ranked)+1), ranked, color=BLUE, linewidth=1.5)
    plt.axhline(256, color=PURPLE, linestyle="--", label="Limit GraphaFolda: 256/PDB/epokę")
    plt.yscale("log"); plt.xlabel("Ranga grupy PDB"); plt.ylabel("Deskryptory na PDB (skala logarytmiczna)")
    plt.legend(frameon=False)
    savefig(figures / "09_pdb_group_imbalance.png")

    # Figure 10: exact-sequence duplication/ambiguity.
    unique_groups = sum(len(rows) == 1 for rows in sequence_groups.values())
    repeated_consistent = sum(len(rows) > 1 and len({r[3] for r in rows}) == 1 for rows in sequence_groups.values())
    repeated_ambiguous = len(ambiguous_sequence_groups)
    plt.figure(figsize=(8.2, 5.0))
    labels = ["pojedyncze wystąpienie", "powtórzone, zgodne etykiety", "powtórzone, sprzeczne etykiety"]
    values = [unique_groups, repeated_consistent, repeated_ambiguous]
    bars = plt.bar(labels, values, color=(LIGHT_BLUE, BLUE, PURPLE))
    plt.ylabel("Liczba odrębnych grup sekwencji")
    plt.xticks(rotation=12, ha="right")
    for bar, value in zip(bars, values):
        plt.text(bar.get_x()+bar.get_width()/2, value, format_count(value), ha="center", va="bottom")
    savefig(figures / "10_sequence_duplication.png")

    # Figure 11: dimensional alignment of IDX/AMT/CMT.
    plt.figure(figsize=(7.8, 4.9))
    alignment_values = [alignment["all_three_equal"], alignment["dimension_mismatch"]]
    bars = plt.bar(["IDX = AMT = CMT", "niezgodne wymiary"], alignment_values,
                   color=[BLUE, PURPLE])
    plt.yscale("log"); plt.ylabel("Liczba deskryptorów (skala logarytmiczna)")
    for bar, value in zip(bars, alignment_values):
        plt.text(bar.get_x()+bar.get_width()/2, value, format_count(value), ha="center", va="bottom")
    savefig(figures / "11_dimension_alignment.png")

    statistics = {
        "scope": "current TrainingSet/split.csv reconstruction",
        "manifest_rows": len(manifest), "SDT_builder_eligible_reconstructed": complete_sdt,
        "complete_idx_amt_cmt": complete_triples,
        "SDT_builder_eligible_by_split": dict(sdt_records),
        "availability": {str(key): value for key, value in availability.items()},
        "dimension_alignment": dict(alignment),
        "records_by_split": split_records, "PDB_groups_by_split": split_groups,
        "PDB_group_overlap_train_val": len(cross_split_groups),
        "length_quantiles": {split: quantiles(lengths[split]) for split in ("train", "val")},
        "raw_AMT_by_split": raw_by_split,
        "raw_AMT_on_SDT_builder_eligible_subset": sdt_raw_by_split,
        "GraphaFold_normalized_by_split_recomputed": {split: dict(normalized[split]) for split in ("train", "val")},
        "GraphaFold_normalized_with_load_graph_max_size": {
            split: dict(normalized_training_semantics[split]) for split in ("train", "val")
        },
        "candidate_counts_on_dimension_mismatches_only": {
            policy: {split: dict(counts) for split, counts in by_split.items()}
            for policy, by_split in candidate_semantics.items()
        },
        "reclassified_raw_gt1": {split: dict(reclassified[split]) for split in ("train", "val")},
        "promoted_raw_label1": {split: dict(promoted[split]) for split in ("train", "val")},
        "sequence_duplication": {
            "distinct_sequences": len(sequence_groups),
            "duplicate_sequence_groups": len(duplicate_sequence_groups),
            "ambiguous_sequence_groups": len(ambiguous_sequence_groups),
            "samples_in_ambiguous_groups": sum(len(rows) for rows in ambiguous_sequence_groups.values()),
            "cross_split_exact_sequence_groups": len(cross_split_sequences),
            "cross_split_ambiguous_sequence_groups": len(cross_split_ambiguous),
            "cross_split_samples": sum(len(rows) for rows in cross_split_sequences.values()),
        },
        "group_balance": {
            "max_descriptors_per_PDB": max(ranked), "median_descriptors_per_PDB": float(np.median(ranked)),
            "groups_above_256": sum(value > 256 for value in ranked),
            "GraphaFold_train_samples_per_epoch_at_cap256": sum(min(value, 256) for value in groups["train"].values()),
        },
        "authoritative_GraphaFold_audit": audit,
    }
    (args.output_dir / "statistics.json").write_text(json.dumps(statistics, indent=2) + "\n")
    print(json.dumps({key: statistics[key] for key in (
        "manifest_rows", "SDT_builder_eligible_reconstructed", "complete_idx_amt_cmt",
        "records_by_split", "PDB_groups_by_split", "dimension_alignment",
        "length_quantiles", "sequence_duplication", "group_balance")}, indent=2))


if __name__ == "__main__":
    main()

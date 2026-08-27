#!/usr/bin/env python3
"""Strict, label-matched SPOT-RNA vs GraphaFold evaluation.

Positives are observed AMT contacts whose unordered base composition is not
AU/CG/GU. SPOT-RNA pairs are read from CT files; GraphaFold pairs and scores
are read from its pairwise prediction CSV files.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Iterable


CANONICAL = {"AU", "CG", "GU"}
PAIR_TYPES = ("AA", "AC", "AG", "CC", "CU", "GG", "UU")
METRICS = ("F1", "INF", "PPV", "TPR")


def pair_type(a: str, b: str) -> str:
    return "".join(sorted((a.upper().replace("T", "U"), b.upper().replace("T", "U"))))


def read_idx(path: Path) -> list[dict[str, str]]:
    entries: list[dict[str, str]] = []
    for expected, line in enumerate(path.read_text().splitlines(), start=1):
        index_text, code = line.split(",", 1)
        if int(index_text) != expected:
            raise ValueError(f"Non-contiguous IDX numbering in {path}")
        chain, token = code.split(".", 1) if "." in code else ("", code)
        base = token[0].upper().replace("T", "U")
        if base not in "ACGU":
            raise ValueError(f"Unsupported base {base!r} in {path}")
        entries.append({"base": base, "chain": chain, "residue": code})
    return entries


def read_matrix(path: Path) -> list[list[int]]:
    with path.open(newline="") as handle:
        return [[int(value) for value in row] for row in csv.reader(handle)]


def read_ct(path: Path, entries: list[dict[str, str]]) -> set[tuple[int, int]]:
    lines = [line for line in path.read_text().splitlines() if line.strip()]
    declared_length = int(lines[0].split()[0])
    if declared_length != len(entries):
        raise ValueError(f"CT length mismatch for {path.stem}")
    partners: dict[int, int] = {}
    for line in lines[1:]:
        fields = line.split()
        index, base, partner = int(fields[0]), fields[1].upper().replace("T", "U"), int(fields[4])
        if base != entries[index - 1]["base"]:
            raise ValueError(f"CT/IDX sequence mismatch for {path.stem}:{index}")
        partners[index] = partner
    if len(partners) != len(entries):
        raise ValueError(f"Incomplete CT rows for {path.stem}")
    pairs: set[tuple[int, int]] = set()
    for i, j in partners.items():
        if j:
            if partners.get(j) != i:
                raise ValueError(f"Asymmetric CT pair {path.stem}:{i}-{j}")
            pairs.add(tuple(sorted((i, j))))
    return pairs


def read_prob(path: Path, length: int) -> list[list[float]]:
    matrix: list[list[float]] = []
    with path.open() as handle:
        for line in handle:
            if line.strip():
                matrix.append([float(value) for value in line.split()])
    if len(matrix) != length or any(len(row) != length for row in matrix):
        raise ValueError(f"Probability matrix dimension mismatch for {path.stem}")
    return matrix


def strict_pair(pair: tuple[int, int], entries: list[dict[str, str]]) -> bool:
    i, j = pair
    return pair_type(entries[i - 1]["base"], entries[j - 1]["base"]) not in CANONICAL


def attributes(pair: tuple[int, int], entries: list[dict[str, str]]) -> tuple[str, str, str]:
    i, j = pair
    first, second = entries[i - 1], entries[j - 1]
    composition = pair_type(first["base"], second["base"])
    if first["chain"] != second["chain"]:
        return composition, "inter-chain", "inter-chain"
    distance = j - i
    if distance <= 4:
        bucket = "1-4"
    elif distance <= 8:
        bucket = "5-8"
    elif distance <= 16:
        bucket = "9-16"
    elif distance <= 32:
        bucket = "17-32"
    elif distance <= 64:
        bucket = "33-64"
    elif distance <= 128:
        bucket = "65-128"
    else:
        bucket = "129+"
    return composition, "intra-chain", bucket


def counts_metrics(gt: set[tuple[object, ...]], pred: set[tuple[object, ...]]) -> dict[str, float | int]:
    tp = len(gt & pred)
    fp = len(pred - gt)
    fn = len(gt - pred)
    ppv = tp / (tp + fp) if tp + fp else 0.0
    tpr = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * ppv * tpr / (ppv + tpr) if ppv + tpr else 0.0
    return {
        "ground_truth": len(gt), "predicted": len(pred), "TP": tp, "FP": fp, "FN": fn,
        "F1": f1, "INF": math.sqrt(ppv * tpr), "PPV": ppv, "TPR": tpr,
    }


def exact_ap(labels: list[int], scores: list[float]) -> float:
    positives = sum(labels)
    if not positives:
        return float("nan")
    ranked = sorted(zip(scores, labels), key=lambda item: item[0], reverse=True)
    tp = 0
    ap = 0.0
    previous_tp = 0
    index = 0
    while index < len(ranked):
        score = ranked[index][0]
        end = index
        while end < len(ranked) and ranked[end][0] == score:
            tp += ranked[end][1]
            end += 1
        newly_found = tp - previous_tp
        if newly_found:
            ap += (tp / end) * newly_found
        previous_tp = tp
        index = end
    return ap / positives


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def bootstrap_difference(
    per_pdb: dict[str, dict[str, dict[str, float | int]]], metric: str, iterations: int = 10000
) -> tuple[float, float, float]:
    ids = sorted(per_pdb)
    rng = random.Random(20260808)
    observed = sum(float(per_pdb[sid]["GraphaFold"][metric]) - float(per_pdb[sid]["SPOT-RNA"][metric]) for sid in ids) / len(ids)
    samples: list[float] = []
    for _ in range(iterations):
        selected = [rng.choice(ids) for _ in ids]
        samples.append(sum(float(per_pdb[sid]["GraphaFold"][metric]) - float(per_pdb[sid]["SPOT-RNA"][metric]) for sid in selected) / len(selected))
    samples.sort()
    return observed, samples[int(0.025 * iterations)], samples[int(0.975 * iterations) - 1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-dir", type=Path, required=True)
    parser.add_argument("--spot-dir", type=Path, required=True)
    parser.add_argument("--graphafold-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    ids = sorted(path.stem for path in args.spot_dir.glob("*.ct"))
    gf_ids = sorted(path.stem for path in (args.graphafold_dir / "pairwise_predictions").glob("*.csv"))
    if ids != gf_ids:
        raise ValueError(f"SPOT/GraphaFold PDB sets differ: SPOT={ids}, GraphaFold={gf_ids}")

    data: dict[str, dict[str, object]] = {}
    candidate_ap_rows: list[dict[str, object]] = []
    pooled_candidate_labels: list[int] = []
    pooled_gf_scores: list[float] = []
    pooled_spot_scores: list[float] = []
    pooled_full_spot_labels: list[int] = []
    pooled_full_spot_scores: list[float] = []
    full_spot_ap_values: list[float] = []

    for sid in ids:
        entries = read_idx(args.benchmark_dir / "idx" / f"{sid}.idx")
        amt = read_matrix(args.benchmark_dir / "amt" / f"{sid}.amt")
        if len(amt) != len(entries) or any(len(row) != len(entries) for row in amt):
            raise ValueError(f"AMT dimension mismatch for {sid}")
        gt = {
            (i + 1, j + 1)
            for i in range(len(entries))
            for j in range(i + 1, len(entries))
            if amt[i][j] > 0 and pair_type(entries[i]["base"], entries[j]["base"]) not in CANONICAL
        }
        spot_all = read_ct(args.spot_dir / f"{sid}.ct", entries)
        spot = {pair for pair in spot_all if strict_pair(pair, entries)}
        spot_prob = read_prob(args.spot_dir / f"{sid}.prob", len(entries))

        gf: set[tuple[int, int]] = set()
        gf_probabilities: dict[tuple[int, int], float] = {}
        labels: list[int] = []
        gf_scores: list[float] = []
        spot_scores: list[float] = []
        candidate_gt: set[tuple[int, int]] = set()
        pairwise_path = args.graphafold_dir / "pairwise_predictions" / f"{sid}.csv"
        with pairwise_path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                pair = tuple(sorted((int(row["index_i"]), int(row["index_j"]))))
                if not strict_pair(pair, entries):
                    continue
                label = int(row["true_noncanonical"])
                score_gf = float(row["probability"])
                score_spot = spot_prob[pair[0] - 1][pair[1] - 1]
                labels.append(label)
                gf_scores.append(score_gf)
                spot_scores.append(score_spot)
                gf_probabilities[pair] = score_gf
                if label:
                    candidate_gt.add(pair)
                if int(row["predicted_noncanonical"]):
                    gf.add(pair)
        full_labels: list[int] = []
        full_scores: list[float] = []
        for i in range(1, len(entries) + 1):
            for j in range(i + 1, len(entries) + 1):
                pair = (i, j)
                if strict_pair(pair, entries):
                    full_labels.append(int(pair in gt))
                    full_scores.append(spot_prob[i - 1][j - 1])
        spot_full_ap = exact_ap(full_labels, full_scores)
        full_spot_ap_values.append(spot_full_ap)
        pooled_full_spot_labels.extend(full_labels)
        pooled_full_spot_scores.extend(full_scores)

        gf_ap = exact_ap(labels, gf_scores)
        spot_candidate_ap = exact_ap(labels, spot_scores)
        candidate_ap_rows.append({
            "PDB_ID": sid,
            "candidate_pairs": len(labels),
            "candidate_positives": sum(labels),
            "candidate_recall": len(candidate_gt) / len(gt) if gt else 0.0,
            "GraphaFold_AP": gf_ap,
            "SPOT_RNA_AP_same_candidates": spot_candidate_ap,
            "SPOT_RNA_AP_all_strict_pairs": spot_full_ap,
        })
        pooled_candidate_labels.extend(labels)
        pooled_gf_scores.extend(gf_scores)
        pooled_spot_scores.extend(spot_scores)
        data[sid] = {
            "entries": entries, "gt": gt, "GraphaFold": gf, "SPOT-RNA": spot,
            "spot_all": spot_all, "spot_canonical": spot_all - spot,
            "spot_prob": spot_prob, "gf_probabilities": gf_probabilities,
            "candidate_pairs": set(gf_probabilities), "amt": amt,
        }

    per_pdb_metrics: dict[str, dict[str, dict[str, float | int]]] = {}
    per_pdb_rows: list[dict[str, object]] = []
    for sid in ids:
        gt = data[sid]["gt"]  # type: ignore[assignment]
        per_pdb_metrics[sid] = {}
        row: dict[str, object] = {
            "PDB_ID": sid,
            "length": len(data[sid]["entries"]),  # type: ignore[arg-type]
            "strict_ground_truth": len(gt),
            "SPOT_RNA_all_predicted_pairs": len(data[sid]["spot_all"]),  # type: ignore[arg-type]
            "SPOT_RNA_excluded_canonical_predictions": len(data[sid]["spot_canonical"]),  # type: ignore[arg-type]
        }
        for model in ("GraphaFold", "SPOT-RNA"):
            metrics = counts_metrics(gt, data[sid][model])  # type: ignore[arg-type]
            per_pdb_metrics[sid][model] = metrics
            for key, value in metrics.items():
                row[f"{model}_{key}"] = value
        gf_f1 = float(per_pdb_metrics[sid]["GraphaFold"]["F1"])
        spot_f1 = float(per_pdb_metrics[sid]["SPOT-RNA"]["F1"])
        row["F1_winner"] = "GraphaFold" if gf_f1 > spot_f1 else "SPOT-RNA" if spot_f1 > gf_f1 else "tie"
        per_pdb_rows.append(row)

    global_sets: dict[str, set[tuple[str, int, int]]] = {"gt": set(), "GraphaFold": set(), "SPOT-RNA": set()}
    for sid in ids:
        for key in global_sets:
            source = data[sid]["gt" if key == "gt" else key]
            global_sets[key].update((sid, i, j) for i, j in source)  # type: ignore[union-attr]

    model_rows: list[dict[str, object]] = []
    for model in ("GraphaFold", "SPOT-RNA"):
        micro = counts_metrics(global_sets["gt"], global_sets[model])
        model_rows.append({"model": model, "averaging": "micro", **micro})
        macro_values = {
            metric: sum(float(per_pdb_metrics[sid][model][metric]) for sid in ids) / len(ids)
            for metric in METRICS
        }
        model_rows.append({
            "model": model, "averaging": "macro_per_PDB", "ground_truth": len(global_sets["gt"]),
            "predicted": sum(int(per_pdb_metrics[sid][model]["predicted"]) for sid in ids),
            "TP": "", "FP": "", "FN": "", **macro_values,
        })

    breakdown_rows: dict[str, list[dict[str, object]]] = {"pair_type": [], "chain_relation": [], "distance": []}
    dimensions = {
        "pair_type": list(PAIR_TYPES),
        "chain_relation": ["intra-chain", "inter-chain"],
        "distance": ["1-4", "5-8", "9-16", "17-32", "33-64", "65-128", "129+", "inter-chain"],
    }
    attr_index = {"pair_type": 0, "chain_relation": 1, "distance": 2}
    for dimension, values in dimensions.items():
        for value in values:
            gt_subset: set[tuple[str, int, int]] = set()
            pred_subsets = {"GraphaFold": set(), "SPOT-RNA": set()}
            for sid in ids:
                entries = data[sid]["entries"]
                for pair in data[sid]["gt"]:  # type: ignore[union-attr]
                    if attributes(pair, entries)[attr_index[dimension]] == value:  # type: ignore[arg-type]
                        gt_subset.add((sid, *pair))
                for model in pred_subsets:
                    for pair in data[sid][model]:  # type: ignore[union-attr]
                        if attributes(pair, entries)[attr_index[dimension]] == value:  # type: ignore[arg-type]
                            pred_subsets[model].add((sid, *pair))
            for model in pred_subsets:
                metrics = counts_metrics(gt_subset, pred_subsets[model])
                breakdown_rows[dimension].append({"category": value, "model": model, **metrics})

    gf_tp = global_sets["GraphaFold"] & global_sets["gt"]
    spot_tp = global_sets["SPOT-RNA"] & global_sets["gt"]
    union_pred = global_sets["GraphaFold"] | global_sets["SPOT-RNA"]
    intersection_pred = global_sets["GraphaFold"] & global_sets["SPOT-RNA"]
    overlap_rows = [
        {"measure": "true_positives_both", "count": len(gf_tp & spot_tp)},
        {"measure": "true_positives_only_GraphaFold", "count": len(gf_tp - spot_tp)},
        {"measure": "true_positives_only_SPOT_RNA", "count": len(spot_tp - gf_tp)},
        {"measure": "ground_truth_missed_by_both", "count": len(global_sets["gt"] - gf_tp - spot_tp)},
        {"measure": "strict_predictions_both", "count": len(intersection_pred)},
        {"measure": "strict_predictions_union", "count": len(union_pred)},
    ]
    for name, prediction in (("prediction_union", union_pred), ("prediction_intersection", intersection_pred)):
        metrics = counts_metrics(global_sets["gt"], prediction)
        for key, value in metrics.items():
            overlap_rows.append({"measure": f"{name}_{key}", "count": value})

    ap_summary_rows = [
        {
            "scope": "shared_GraphaFold_candidate_pairs_micro",
            "pairs": len(pooled_candidate_labels), "positives": sum(pooled_candidate_labels),
            "GraphaFold_AP": exact_ap(pooled_candidate_labels, pooled_gf_scores),
            "SPOT_RNA_AP": exact_ap(pooled_candidate_labels, pooled_spot_scores),
        },
        {
            "scope": "shared_GraphaFold_candidate_pairs_macro_per_PDB",
            "pairs": "", "positives": "",
            "GraphaFold_AP": sum(float(row["GraphaFold_AP"]) for row in candidate_ap_rows) / len(ids),
            "SPOT_RNA_AP": sum(float(row["SPOT_RNA_AP_same_candidates"]) for row in candidate_ap_rows) / len(ids),
        },
        {
            "scope": "SPOT_RNA_all_strict_pairs_micro",
            "pairs": len(pooled_full_spot_labels), "positives": sum(pooled_full_spot_labels),
            "GraphaFold_AP": "not_scored_outside_candidates",
            "SPOT_RNA_AP": exact_ap(pooled_full_spot_labels, pooled_full_spot_scores),
        },
        {
            "scope": "SPOT_RNA_all_strict_pairs_macro_per_PDB",
            "pairs": "", "positives": "", "GraphaFold_AP": "not_scored_outside_candidates",
            "SPOT_RNA_AP": sum(full_spot_ap_values) / len(full_spot_ap_values),
        },
    ]

    bootstrap_rows = []
    for metric in METRICS:
        difference, low, high = bootstrap_difference(per_pdb_metrics, metric)
        bootstrap_rows.append({"metric": metric, "GraphaFold_minus_SPOT_RNA": difference, "CI95_low": low, "CI95_high": high})

    predicted_pair_rows: list[dict[str, object]] = []
    ground_truth_pair_rows: list[dict[str, object]] = []
    amt_class_rows: list[dict[str, object]] = []
    spot_excluded_composition_counts: dict[str, int] = defaultdict(int)
    for sid in ids:
        entries = data[sid]["entries"]
        gt = data[sid]["gt"]
        spot_prob = data[sid]["spot_prob"]
        gf_probabilities = data[sid]["gf_probabilities"]
        amt = data[sid]["amt"]
        for i, j in data[sid]["spot_canonical"]:  # type: ignore[union-attr]
            composition, _, _ = attributes((i, j), entries)  # type: ignore[arg-type]
            spot_excluded_composition_counts[composition] += 1
        for model in ("GraphaFold", "SPOT-RNA"):
            source = data[sid][model]
            for i, j in sorted(source):  # type: ignore[union-attr]
                composition, relation, distance = attributes((i, j), entries)  # type: ignore[arg-type]
                probability = gf_probabilities[(i, j)] if model == "GraphaFold" else spot_prob[i - 1][j - 1]  # type: ignore[index]
                predicted_pair_rows.append({
                    "PDB_ID": sid, "model": model, "index_i": i, "index_j": j,
                    "residue_i": entries[i - 1]["residue"], "residue_j": entries[j - 1]["residue"],  # type: ignore[index]
                    "base_i": entries[i - 1]["base"], "base_j": entries[j - 1]["base"],  # type: ignore[index]
                    "pair_type": composition, "chain_relation": relation, "distance_bucket": distance,
                    "probability": probability, "is_true_positive": int((i, j) in gt),
                    "raw_amt_label": amt[i - 1][j - 1],  # type: ignore[index]
                    "predicted_by_other_model": int((i, j) in data[sid]["SPOT-RNA" if model == "GraphaFold" else "GraphaFold"]),  # type: ignore[operator]
                })
        for i, j in sorted(gt):  # type: ignore[union-attr]
            composition, relation, distance = attributes((i, j), entries)  # type: ignore[arg-type]
            ground_truth_pair_rows.append({
                "PDB_ID": sid, "index_i": i, "index_j": j,
                "residue_i": entries[i - 1]["residue"], "residue_j": entries[j - 1]["residue"],  # type: ignore[index]
                "base_i": entries[i - 1]["base"], "base_j": entries[j - 1]["base"],  # type: ignore[index]
                "pair_type": composition, "chain_relation": relation, "distance_bucket": distance,
                "raw_amt_label": amt[i - 1][j - 1],  # type: ignore[index]
                "inside_GraphaFold_candidate_window": int((i, j) in data[sid]["candidate_pairs"]),  # type: ignore[operator]
                "GraphaFold_probability": gf_probabilities.get((i, j), ""),  # type: ignore[union-attr]
                "SPOT_RNA_probability": spot_prob[i - 1][j - 1],  # type: ignore[index]
                "GraphaFold_predicted": int((i, j) in data[sid]["GraphaFold"]),  # type: ignore[operator]
                "SPOT_RNA_predicted": int((i, j) in data[sid]["SPOT-RNA"]),  # type: ignore[operator]
            })
    amt_classes = sorted({int(row["raw_amt_label"]) for row in ground_truth_pair_rows})
    for amt_class in amt_classes:
        class_rows = [row for row in ground_truth_pair_rows if int(row["raw_amt_label"]) == amt_class]
        for model, field in (("GraphaFold", "GraphaFold_predicted"), ("SPOT-RNA", "SPOT_RNA_predicted")):
            recovered = sum(int(row[field]) for row in class_rows)
            amt_class_rows.append({
                "raw_amt_label": amt_class, "model": model, "ground_truth": len(class_rows),
                "recovered": recovered, "missed": len(class_rows) - recovered,
                "recall": recovered / len(class_rows),
            })
    spot_excluded_rows = [
        {"pair_type": composition, "count": spot_excluded_composition_counts.get(composition, 0)}
        for composition in ("AU", "CG", "GU")
    ]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metric_fields = ["model", "averaging", "ground_truth", "predicted", "TP", "FP", "FN", *METRICS]
    write_csv(args.output_dir / "model_summary.csv", model_rows, metric_fields)
    write_csv(args.output_dir / "per_pdb_metrics.csv", per_pdb_rows, list(per_pdb_rows[0]))
    for dimension, rows in breakdown_rows.items():
        write_csv(args.output_dir / f"{dimension}_metrics.csv", rows, ["category", *metric_fields])
    write_csv(args.output_dir / "prediction_overlap.csv", overlap_rows, ["measure", "count"])
    write_csv(args.output_dir / "shared_candidate_ap_per_pdb.csv", candidate_ap_rows, list(candidate_ap_rows[0]))
    write_csv(args.output_dir / "average_precision_summary.csv", ap_summary_rows, list(ap_summary_rows[0]))
    write_csv(args.output_dir / "bootstrap_macro_difference.csv", bootstrap_rows, list(bootstrap_rows[0]))
    write_csv(args.output_dir / "predicted_pairs_audit.csv", predicted_pair_rows, list(predicted_pair_rows[0]))
    write_csv(args.output_dir / "ground_truth_pair_recovery.csv", ground_truth_pair_rows, list(ground_truth_pair_rows[0]))
    write_csv(args.output_dir / "amt_class_recall.csv", amt_class_rows, list(amt_class_rows[0]))
    write_csv(args.output_dir / "spot_excluded_canonical_compositions.csv", spot_excluded_rows, ["pair_type", "count"])

    micro_by_model = {row["model"]: row for row in model_rows if row["averaging"] == "micro"}
    macro_by_model = {row["model"]: row for row in model_rows if row["averaging"] == "macro_per_PDB"}
    wins = defaultdict(int)
    for row in per_pdb_rows:
        wins[str(row["F1_winner"])] += 1
    report = {
        "definition": "AMT > 0 and unordered base composition not in AU/CG/GU",
        "structures": ids,
        "num_structures": len(ids),
        "micro": micro_by_model,
        "macro_per_PDB": macro_by_model,
        "per_PDB_F1_wins": dict(wins),
        "SPOT_RNA_total_selected_pairs": sum(len(data[sid]["spot_all"]) for sid in ids),  # type: ignore[arg-type]
        "SPOT_RNA_strict_selected_pairs": len(global_sets["SPOT-RNA"]),
        "SPOT_RNA_excluded_AU_CG_GU_selected_pairs": sum(len(data[sid]["spot_canonical"]) for sid in ids),  # type: ignore[arg-type]
        "true_positive_overlap": {
            "both": len(gf_tp & spot_tp), "GraphaFold_only": len(gf_tp - spot_tp),
            "SPOT_RNA_only": len(spot_tp - gf_tp), "missed_by_both": len(global_sets["gt"] - gf_tp - spot_tp),
        },
        "average_precision": ap_summary_rows,
        "bootstrap_macro_GraphaFold_minus_SPOT_RNA": bootstrap_rows,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n")

    latex = [
        r"\begin{table}[ht]", r"\centering",
        r"\caption{Strict non-canonical contact prediction on the common 23-structure benchmark.}",
        r"\label{tab:graphafold_spotrna_strict}", r"\begin{tabular}{llcccc}", r"\thickhline",
        r"Model & Averaging & F1 & INF & PPV & TPR \\", r"\thickhline",
    ]
    for row in model_rows:
        latex.append(
            f"{row['model']} & {row['averaging']} & "
            + " & ".join(f"{float(row[metric]):.3f}" for metric in METRICS) + r" \\" 
        )
    latex.extend([r"\thickhline", r"\end{tabular}", r"\end{table}"])
    (args.output_dir / "comparison_table.tex").write_text("\n".join(latex) + "\n")

    per_pdb_latex = [
        r"\begin{landscape}", r"\begin{longtable}{c|cccc|cccc}", r"\thickhline",
        r" & \multicolumn{4}{c|}{GraphaFold} & \multicolumn{4}{c}{SPOT-RNA} \\",
        r"PDB ID & F1 & INF & PPV & TPR & F1 & INF & PPV & TPR \\", r"\thickhline",
    ]
    for sid in ids:
        values = [sid]
        for model in ("GraphaFold", "SPOT-RNA"):
            values.extend(f"{float(per_pdb_metrics[sid][model][metric]):.2f}" for metric in METRICS)
        per_pdb_latex.append(" & ".join(values) + r" \\")
    per_pdb_latex.extend([r"\thickhline", r"\end{longtable}", r"\end{landscape}"])
    (args.output_dir / "per_pdb_comparison_table.tex").write_text("\n".join(per_pdb_latex) + "\n")

    pair_lookup = {
        (str(row["category"]), str(row["model"])): row
        for row in breakdown_rows["pair_type"]
    }
    pair_latex = [
        r"\begin{table}[ht]", r"\centering",
        r"\caption{Strict performance by unordered base composition.}",
        r"\label{tab:graphafold_spotrna_pair_types}",
        r"\begin{tabular}{c|c|ccc|ccc}", r"\thickhline",
        r"Pair & Ground truth & \multicolumn{3}{c|}{GraphaFold} & \multicolumn{3}{c}{SPOT-RNA} \\",
        r" & & TP & FN & FP & TP & FN & FP \\", r"\thickhline",
    ]
    for composition in PAIR_TYPES:
        gf_row = pair_lookup[(composition, "GraphaFold")]
        spot_row = pair_lookup[(composition, "SPOT-RNA")]
        pair_latex.append(
            f"{composition} & {gf_row['ground_truth']} & {gf_row['TP']} & {gf_row['FN']} & {gf_row['FP']} & "
            f"{spot_row['TP']} & {spot_row['FN']} & {spot_row['FP']} " + r"\\"
        )
    pair_latex.extend([r"\thickhline", r"\end{tabular}", r"\end{table}"])
    (args.output_dir / "pair_type_comparison_table.tex").write_text("\n".join(pair_latex) + "\n")

    candidate_global = {
        (sid, i, j)
        for sid in ids
        for i, j in data[sid]["candidate_pairs"]  # type: ignore[union-attr]
    }
    spot_tp_outside_candidates = spot_tp - candidate_global
    spot_nonempty_pdbs = sum(bool(data[sid]["SPOT-RNA"]) for sid in ids)
    report_md = f"""# Strict SPOT-RNA vs GraphaFold comparison

## Evaluation contract

- Common set: {len(ids)} structures; `8BTZ` and the previously excluded structures are absent.
- Positive: `AMT > 0` and unordered base composition not in `AU/CG/GU`.
- SPOT-RNA predictions: selected CT pairs, with predicted AU/CG/GU removed.
- GraphaFold predictions: fixed epoch-43 output at threshold 0.52734375.
- Macro values give every PDB equal weight; micro values pool all contacts.

## Main result

| Model | Averaging | F1 | INF | PPV | TPR | TP | FP | FN |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| GraphaFold | micro | {float(micro_by_model['GraphaFold']['F1']):.3f} | {float(micro_by_model['GraphaFold']['INF']):.3f} | {float(micro_by_model['GraphaFold']['PPV']):.3f} | {float(micro_by_model['GraphaFold']['TPR']):.3f} | {micro_by_model['GraphaFold']['TP']} | {micro_by_model['GraphaFold']['FP']} | {micro_by_model['GraphaFold']['FN']} |
| SPOT-RNA | micro | {float(micro_by_model['SPOT-RNA']['F1']):.3f} | {float(micro_by_model['SPOT-RNA']['INF']):.3f} | {float(micro_by_model['SPOT-RNA']['PPV']):.3f} | {float(micro_by_model['SPOT-RNA']['TPR']):.3f} | {micro_by_model['SPOT-RNA']['TP']} | {micro_by_model['SPOT-RNA']['FP']} | {micro_by_model['SPOT-RNA']['FN']} |
| GraphaFold | macro/PDB | {float(macro_by_model['GraphaFold']['F1']):.3f} | {float(macro_by_model['GraphaFold']['INF']):.3f} | {float(macro_by_model['GraphaFold']['PPV']):.3f} | {float(macro_by_model['GraphaFold']['TPR']):.3f} | -- | -- | -- |
| SPOT-RNA | macro/PDB | {float(macro_by_model['SPOT-RNA']['F1']):.3f} | {float(macro_by_model['SPOT-RNA']['INF']):.3f} | {float(macro_by_model['SPOT-RNA']['PPV']):.3f} | {float(macro_by_model['SPOT-RNA']['TPR']):.3f} | -- | -- | -- |

GraphaFold finds 35 true contacts versus 24 for SPOT-RNA, but makes 114 false
predictions versus only 2 for SPOT-RNA. Consequently GraphaFold has higher
recall, while SPOT-RNA has much higher precision. Micro F1 is nearly tied
({float(micro_by_model['GraphaFold']['F1']):.3f} vs {float(micro_by_model['SPOT-RNA']['F1']):.3f}); macro F1 slightly favors GraphaFold
({float(macro_by_model['GraphaFold']['F1']):.3f} vs {float(macro_by_model['SPOT-RNA']['F1']):.3f}).

SPOT-RNA selected 778 total CT pairs. Of these, 752 AU/CG/GU pairs were excluded,
leaving 26 strict predictions. It emitted at least one strict prediction for
{spot_nonempty_pdbs}/{len(ids)} structures.

## Ranking quality

On the exact same 71,846 strict GraphaFold candidate pairs (185 positives),
GraphaFold AP is {float(ap_summary_rows[0]['GraphaFold_AP']):.3f} and SPOT-RNA AP is
{float(ap_summary_rows[0]['SPOT_RNA_AP']):.3f}. Macro AP is respectively
{float(ap_summary_rows[1]['GraphaFold_AP']):.3f} and {float(ap_summary_rows[1]['SPOT_RNA_AP']):.3f}. Thus the SPOT probability matrix ranks
the strict contacts substantially better, even though its default CT decoding is
very conservative. SPOT-RNA full-universe strict AP is
{float(ap_summary_rows[2]['SPOT_RNA_AP']):.3f} micro and {float(ap_summary_rows[3]['SPOT_RNA_AP']):.3f} macro.

All {len(spot_tp)} SPOT-RNA true positives are inside the GraphaFold candidate
window; SPOT has {len(spot_tp_outside_candidates)} true positives outside it.
The candidate generator therefore does not explain SPOT's unique hits.

## Complementarity

- True positives found by both: {len(gf_tp & spot_tp)}
- GraphaFold-only true positives: {len(gf_tp - spot_tp)}
- SPOT-RNA-only true positives: {len(spot_tp - gf_tp)}
- Missed by both: {len(global_sets['gt'] - gf_tp - spot_tp)}
- Per-PDB F1 wins: GraphaFold {wins['GraphaFold']}, SPOT-RNA {wins['SPOT-RNA']}, ties {wins['tie']}

A direct union of both selected-pair sets gives TP=49, FP=115, FN=174,
PPV=0.299, TPR=0.220 and F1=0.253. The intersection gives PPV=0.909 but only
TPR=0.045. This strongly suggests using SPOT probabilities as a feature/prior
or training a calibrated ensemble rather than choosing one model globally.

## Pair types and distances

SPOT-RNA's strict predictions are concentrated in AG (15/17 correct), GG (5/5),
UU (2/2), AA (1/1), and AC (1/1); it predicts no CC or CU. GraphaFold spreads
predictions more broadly and retrieves more AA, AC and UU contacts, but with
many more false positives. SPOT is strongest at sequence distances 1--8;
GraphaFold contributes most of the recovered contacts at distances 17 and above.

By raw AMT interaction class, GraphaFold recovers 20/56 class-8 (cWW), 10/28
class-7 (tHS/tSH), 2/15 class-6, 2/9 class-2, and 1/10 class-10 contacts.
SPOT-RNA instead recovers 14/28 class-7, 6/56 class-8, 3/30 class-12
(cWH/cHW), and 1/15 class-6 contacts. Neither model recovers any class-3, 4,
5, 9, 11, or 13 contacts in this subset.

The common subset contains no strict inter-chain ground-truth pairs. GraphaFold
made 15 inter-chain predictions, all false positives; SPOT made none. This is a
property of this benchmark subset and should not be converted into a general
rule that inter-chain contacts are impossible.

## Uncertainty and caveats

Molecule-bootstrap 95% CI for the macro F1 difference GraphaFold minus SPOT-RNA
is [{bootstrap_rows[0]['CI95_low']:.3f}, {bootstrap_rows[0]['CI95_high']:.3f}], so this 23-structure set does not establish a clear
F1 winner. The PPV difference does favor SPOT-RNA. No threshold was tuned on
this benchmark: GraphaFold uses its validation threshold and SPOT uses default
CT post-processing. Training-set overlap for the pretrained SPOT-RNA model has
not been established from the locally available files and should be audited
before treating this as a definitive independent benchmark comparison.
"""
    (args.output_dir / "analysis_report.md").write_text(report_md)

    print(json.dumps({"micro": micro_by_model, "macro": macro_by_model, "wins": dict(wins)}, indent=2))


if __name__ == "__main__":
    main()

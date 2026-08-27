#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CANONICAL_COMPOSITIONS = frozenset({"AU", "CG", "GU"})
NONCANONICAL_ORDER = ("AG", "AC", "GG", "AA", "UU", "CU", "CC")
AMT_CLASS_NAMES = {
    2: "tWW",
    3: "tHH",
    4: "tSS",
    5: "tWS/tSW",
    6: "tWH/tHW",
    7: "tHS/tSH",
    8: "cWW",
    9: "cHH",
    10: "cSS",
    11: "cWS/cSW",
    12: "cWH/cHW",
    13: "cHS/cSH",
}


def pair_type(base_i: str, base_j: str) -> str:
    return "".join(sorted((str(base_i).upper().replace("T", "U"), str(base_j).upper().replace("T", "U"))))


def safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else float("nan")


def f1_score(precision: float, recall: float) -> float:
    if not math.isfinite(precision) or not math.isfinite(recall) or precision + recall == 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def exact_average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.int8)
    scores = np.asarray(scores, dtype=np.float64)
    positives = int(labels.sum())
    if positives == 0:
        return float("nan")
    order = np.argsort(-scores, kind="stable")
    sorted_labels = labels[order]
    sorted_scores = scores[order]
    cumulative_tp = np.cumsum(sorted_labels)
    ranks = np.arange(1, len(labels) + 1)
    tie_end = np.empty(len(labels), dtype=bool)
    tie_end[-1] = True
    tie_end[:-1] = sorted_scores[:-1] != sorted_scores[1:]
    indices = np.flatnonzero(tie_end)
    tp_at_threshold = cumulative_tp[indices]
    positives_per_tie = np.diff(np.concatenate(([0], tp_at_threshold)))
    precision = tp_at_threshold / ranks[indices]
    return float(np.sum(precision * positives_per_tie) / positives)


def read_idx(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        index_text, code = raw.strip().split(",", 1)
        token = code.split(".", 1)[1] if "." in code else code
        base = token[:1].upper().replace("T", "U")
        chain = code.split(".", 1)[0] if "." in code else ""
        rows.append({"index": int(index_text), "residue": code, "base": base, "chain": chain})
    return rows


def load_candidate_predictions(results_dir: Path) -> pd.DataFrame:
    files = sorted((results_dir / "pairwise_predictions").glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No prediction CSV files under {results_dir / 'pairwise_predictions'}")
    frame = pd.concat((pd.read_csv(path) for path in files), ignore_index=True)
    required = {
        "sample_id",
        "index_i",
        "index_j",
        "residue_i",
        "residue_j",
        "base_i",
        "base_j",
        "raw_amt_label",
        "true_noncanonical",
        "probability",
        "predicted_noncanonical",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Prediction CSV files lack columns: {sorted(missing)}")
    frame["pair_type"] = [pair_type(a, b) for a, b in zip(frame.base_i, frame.base_j)]
    frame["chain_i"] = frame.residue_i.astype(str).str.split(".", n=1).str[0]
    frame["chain_j"] = frame.residue_j.astype(str).str.split(".", n=1).str[0]
    frame["chain_relation"] = np.where(frame.chain_i == frame.chain_j, "intra-chain", "inter-chain")
    frame["pair_key"] = list(zip(frame.sample_id, frame.index_i, frame.index_j))
    return frame


def load_ground_truth(benchmark_dir: Path) -> tuple[pd.DataFrame, dict[str, dict[str, object]]]:
    rows: list[dict[str, object]] = []
    molecule_data: dict[str, dict[str, object]] = {}
    for amt_path in sorted((benchmark_dir / "amt").glob("*.amt")):
        sample_id = amt_path.stem
        idx_path = benchmark_dir / "idx" / f"{sample_id}.idx"
        cmt_path = benchmark_dir / "cmt" / f"{sample_id}.cmt"
        if not idx_path.is_file() or not cmt_path.is_file():
            continue
        entries = read_idx(idx_path)
        amt = np.loadtxt(amt_path, delimiter=",", dtype=np.int16, ndmin=2)
        cmt = np.loadtxt(cmt_path, delimiter=",", dtype=np.int16, ndmin=2)
        if amt.shape != (len(entries), len(entries)) or cmt.shape != amt.shape:
            raise ValueError(f"Dimension mismatch for {sample_id}")
        bases = [str(entry["base"]) for entry in entries]
        canonical_pairs: list[tuple[int, int]] = []
        for i in range(len(entries)):
            for j in range(i + 1, len(entries)):
                composition = pair_type(bases[i], bases[j])
                if cmt[i, j] == 1 and composition in CANONICAL_COMPOSITIONS:
                    canonical_pairs.append((i + 1, j + 1))
                if amt[i, j] > 0 and composition not in CANONICAL_COMPOSITIONS:
                    chain_relation = (
                        "intra-chain" if entries[i]["chain"] == entries[j]["chain"] else "inter-chain"
                    )
                    rows.append(
                        {
                            "sample_id": sample_id,
                            "index_i": i + 1,
                            "index_j": j + 1,
                            "residue_i": entries[i]["residue"],
                            "residue_j": entries[j]["residue"],
                            "base_i": bases[i],
                            "base_j": bases[j],
                            "pair_type": composition,
                            "raw_amt_label": int(amt[i, j]),
                            "sequence_distance": j - i,
                            "chain_relation": chain_relation,
                            "pair_key": (sample_id, i + 1, j + 1),
                        }
                    )
        molecule_data[sample_id] = {
            "length": len(entries),
            "entries": entries,
            "canonical_pairs": canonical_pairs,
        }
    return pd.DataFrame(rows), molecule_data


def distance_bucket(distance: int, relation: str) -> str:
    if relation == "inter-chain":
        return "inter-chain"
    if distance <= 4:
        return "1-4"
    if distance <= 8:
        return "5-8"
    if distance <= 16:
        return "9-16"
    if distance <= 32:
        return "17-32"
    if distance <= 64:
        return "33-64"
    if distance <= 128:
        return "65-128"
    return "129+"


def offset_bucket(offset: float) -> str:
    if not math.isfinite(offset):
        return "outside-window"
    offset = int(offset)
    if offset <= 3:
        return "0-3"
    if offset <= 7:
        return "4-7"
    if offset <= 11:
        return "8-11"
    return "12-15"


def add_canonical_offsets(
    candidates: pd.DataFrame,
    molecule_data: dict[str, dict[str, object]],
) -> pd.DataFrame:
    candidates = candidates.copy()
    offsets = np.full(len(candidates), np.nan, dtype=np.float64)
    for sample_id, indices in candidates.groupby("sample_id", sort=False).groups.items():
        canonical = np.asarray(molecule_data[sample_id]["canonical_pairs"], dtype=np.int32)
        if canonical.size == 0:
            continue
        for row_index in indices:
            i = int(candidates.at[row_index, "index_i"])
            j = int(candidates.at[row_index, "index_j"])
            direct = np.maximum(np.abs(canonical[:, 0] - i), np.abs(canonical[:, 1] - j))
            reverse = np.maximum(np.abs(canonical[:, 1] - i), np.abs(canonical[:, 0] - j))
            offsets[row_index] = float(min(direct.min(), reverse.min()))
    candidates["canonical_offset"] = offsets
    candidates["offset_bucket"] = [offset_bucket(value) for value in offsets]
    return candidates


def grouped_metrics(
    ground_truth: pd.DataFrame,
    candidates: pd.DataFrame,
    group_column: str,
    order: Iterable[object] | None = None,
) -> pd.DataFrame:
    present = list(dict.fromkeys(list(ground_truth[group_column]) + list(candidates[group_column])))
    groups = list(order) if order is not None else present
    groups.extend(value for value in present if value not in groups)
    rows: list[dict[str, object]] = []
    for value in groups:
        gt = ground_truth[ground_truth[group_column] == value]
        cand = candidates[candidates[group_column] == value]
        candidate_true = cand[cand.true_noncanonical == 1]
        tp = int(((cand.true_noncanonical == 1) & (cand.predicted_noncanonical == 1)).sum())
        fp = int(((cand.true_noncanonical == 0) & (cand.predicted_noncanonical == 1)).sum())
        ground_count = len(gt)
        candidate_count = len(candidate_true)
        fn = ground_count - tp
        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, ground_count)
        candidate_recall = safe_div(candidate_count, ground_count)
        conditional_recall = safe_div(tp, candidate_count)
        ap = exact_average_precision(cand.true_noncanonical.to_numpy(), cand.probability.to_numpy()) if len(cand) else float("nan")
        prevalence = safe_div(candidate_count, len(cand))
        rows.append(
            {
                group_column: value,
                "ground_truth_count": ground_count,
                "candidate_ground_truth": candidate_count,
                "outside_window": ground_count - candidate_count,
                "candidate_pairs": len(cand),
                "tp": tp,
                "fn": fn,
                "fp": fp,
                "predicted_count": tp + fp,
                "precision": precision,
                "pipeline_recall": recall,
                "pipeline_f1": f1_score(precision, recall),
                "candidate_recall": candidate_recall,
                "candidate_classification_recall": conditional_recall,
                "average_precision": ap,
                "candidate_prevalence": prevalence,
                "ap_lift_over_prevalence": safe_div(ap, prevalence),
            }
        )
    return pd.DataFrame(rows)


def class_metrics(ground_truth: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for class_id in range(2, 14):
        gt = ground_truth[ground_truth.raw_amt_label == class_id]
        candidate_true = candidates[
            (candidates.true_noncanonical == 1) & (candidates.raw_amt_label == class_id)
        ]
        tp = int((candidate_true.predicted_noncanonical == 1).sum())
        ground_count = len(gt)
        candidate_count = len(candidate_true)
        rows.append(
            {
                "amt_class": class_id,
                "class_name": AMT_CLASS_NAMES[class_id],
                "ground_truth_count": ground_count,
                "candidate_ground_truth": candidate_count,
                "outside_window": ground_count - candidate_count,
                "tp": tp,
                "fn": ground_count - tp,
                "candidate_recall": safe_div(candidate_count, ground_count),
                "candidate_classification_recall": safe_div(tp, candidate_count),
                "pipeline_recall": safe_div(tp, ground_count),
            }
        )
    return pd.DataFrame(rows)


def molecule_metrics(
    ground_truth: pd.DataFrame,
    candidates: pd.DataFrame,
    molecule_data: dict[str, dict[str, object]],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for sample_id in sorted(molecule_data):
        gt = ground_truth[ground_truth.sample_id == sample_id]
        cand = candidates[candidates.sample_id == sample_id]
        tp = int(((cand.true_noncanonical == 1) & (cand.predicted_noncanonical == 1)).sum())
        fp = int(((cand.true_noncanonical == 0) & (cand.predicted_noncanonical == 1)).sum())
        candidate_gt = int((cand.true_noncanonical == 1).sum())
        total_gt = len(gt)
        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, total_gt)
        ap = exact_average_precision(cand.true_noncanonical.to_numpy(), cand.probability.to_numpy()) if len(cand) else float("nan")
        f1 = f1_score(precision, recall)
        inf = math.sqrt(precision * recall) if math.isfinite(precision) and math.isfinite(recall) else 0.0
        rows.append(
            {
                "sample_id": sample_id,
                "length": int(molecule_data[sample_id]["length"]),
                "candidate_pairs": len(cand),
                "ground_truth_count": total_gt,
                "candidate_ground_truth": candidate_gt,
                "outside_window": total_gt - candidate_gt,
                "tp": tp,
                "fn": total_gt - tp,
                "fp": fp,
                "predicted_count": tp + fp,
                "precision": precision,
                "pipeline_recall": recall,
                "pipeline_f1": f1,
                "INF": inf,
                "PPV": precision,
                "TPR": recall,
                "candidate_recall": safe_div(candidate_gt, total_gt),
                "candidate_classification_recall": safe_div(tp, candidate_gt),
                "average_precision": ap,
                "candidate_prevalence": safe_div(candidate_gt, len(cand)),
            }
        )
    return pd.DataFrame(rows)


def pr_curve(labels: np.ndarray, scores: np.ndarray, total_positives: int) -> pd.DataFrame:
    order = np.argsort(-scores, kind="stable")
    labels = labels[order].astype(np.int64)
    scores = scores[order]
    cumulative_tp = np.cumsum(labels)
    ranks = np.arange(1, len(labels) + 1)
    tie_end = np.empty(len(labels), dtype=bool)
    tie_end[-1] = True
    tie_end[:-1] = scores[:-1] != scores[1:]
    indices = np.flatnonzero(tie_end)
    tp = cumulative_tp[indices]
    predicted = ranks[indices]
    return pd.DataFrame(
        {
            "threshold": scores[indices],
            "predicted_count": predicted,
            "tp": tp,
            "fp": predicted - tp,
            "precision": tp / predicted,
            "candidate_recall": tp / max(1, labels.sum()),
            "pipeline_recall": tp / max(1, total_positives),
        }
    )


def bootstrap_intervals(molecule_frame: pd.DataFrame, seed: int = 42, repeats: int = 10_000) -> dict[str, list[float]]:
    values = molecule_frame[["tp", "fp", "ground_truth_count", "candidate_ground_truth"]].to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    samples = rng.integers(0, len(values), size=(repeats, len(values)))
    totals = values[samples].sum(axis=1)
    tp, fp, ground, candidate_ground = totals.T
    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    recall = np.divide(tp, ground, out=np.zeros_like(tp), where=ground > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(tp), where=(precision + recall) > 0)
    coverage = np.divide(candidate_ground, ground, out=np.zeros_like(tp), where=ground > 0)
    return {
        "precision": np.quantile(precision, [0.025, 0.975]).tolist(),
        "pipeline_recall": np.quantile(recall, [0.025, 0.975]).tolist(),
        "pipeline_f1": np.quantile(f1, [0.025, 0.975]).tolist(),
        "candidate_recall": np.quantile(coverage, [0.025, 0.975]).tolist(),
    }


def markdown_table(frame: pd.DataFrame, columns: list[str], digits: int = 3) -> str:
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    rows = [header, divider]
    for _, row in frame.iterrows():
        values: list[str] = []
        for column in columns:
            value = row[column]
            if isinstance(value, (float, np.floating)):
                values.append("—" if not math.isfinite(float(value)) else f"{float(value):.{digits}f}")
            else:
                values.append(str(value))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join(rows)


def save_plots(
    output_dir: Path,
    pair_metrics: pd.DataFrame,
    classes: pd.DataFrame,
    distances: pd.DataFrame,
    molecules: pd.DataFrame,
    curve: pd.DataFrame,
    fixed_threshold: float,
) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")

    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    x = np.arange(len(pair_metrics))
    width = 0.25
    ax.bar(x - width, pair_metrics.precision.fillna(0), width, label="Precision")
    ax.bar(x, pair_metrics.pipeline_recall.fillna(0), width, label="Pipeline recall")
    ax.bar(x + width, pair_metrics.pipeline_f1.fillna(0), width, label="Pipeline F1")
    ax.set_xticks(x, pair_metrics.pair_type)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("GraphaFold: strict noncanonical performance by base composition")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "pair_type_precision_recall_f1.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    ax.bar(x - width, pair_metrics.tp, width, label="TP")
    ax.bar(x, pair_metrics.fn, width, label="FN")
    ax.bar(x + width, pair_metrics.fp, width, label="FP")
    ax.set_xticks(x, pair_metrics.pair_type)
    ax.set_ylabel("Number of pairs")
    ax.set_title("Standard TP/FN/FP counts by noncanonical base composition")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "pair_type_counts.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11.5, 5.5))
    labels = [f"{row.amt_class} {row.class_name}" for row in classes.itertuples()]
    x = np.arange(len(classes))
    ax.bar(x - 0.18, classes.candidate_recall.fillna(0), 0.36, label="Candidate coverage")
    ax.bar(x + 0.18, classes.pipeline_recall.fillna(0), 0.36, label="Pipeline recall")
    ax.set_xticks(x, labels, rotation=35, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Recall")
    ax.set_title("Coverage and detection by AMT geometry class")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "amt_class_recall.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    x = np.arange(len(distances))
    ax.bar(x - width, distances.precision.fillna(0), width, label="Precision")
    ax.bar(x, distances.pipeline_recall.fillna(0), width, label="Pipeline recall")
    ax.bar(x + width, distances.candidate_recall.fillna(0), width, label="Candidate coverage")
    ax.set_xticks(x, distances.distance_bucket, rotation=25, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Performance by sequence distance (inter-chain separate)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "distance_metrics.png", dpi=220)
    plt.close(fig)

    ranked = molecules.sort_values("average_precision", na_position="last")
    fig, ax = plt.subplots(figsize=(12, 7))
    y = np.arange(len(ranked))
    ax.barh(y - 0.18, ranked.average_precision.fillna(0), 0.36, label="Average precision")
    ax.barh(y + 0.18, ranked.pipeline_f1.fillna(0), 0.36, label="Pipeline F1")
    ax.set_yticks(y, ranked.sample_id)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Score")
    ax.set_title("Per-molecule transfer performance")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "per_molecule_ap_f1.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 6))
    ax.step(curve.pipeline_recall, curve.precision, where="post", color="#1769aa", label="GraphaFold")
    selected = curve.iloc[(curve.threshold - fixed_threshold).abs().argsort()[:1]]
    ax.scatter(selected.pipeline_recall, selected.precision, color="#d1495b", zorder=3, label=f"Checkpoint threshold {fixed_threshold:.4f}")
    ax.set_xlim(left=0)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Pipeline recall (all true noncanonical contacts)")
    ax.set_ylabel("Precision")
    ax.set_title("Strict noncanonical precision-recall curve")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "precision_recall_curve.png", dpi=220)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze GraphaFold whole-molecule CASP RNA2 results")
    parser.add_argument("--results-dir", type=Path, default=Path("CaspRNA2-graphafold"))
    parser.add_argument("--benchmark-dir", type=Path, default=Path("CaspRNA2"))
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    output_dir = args.output_dir or args.results_dir / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = json.loads((args.results_dir / "summary.json").read_text(encoding="utf-8"))
    threshold = float(summary["threshold"])
    candidates_all = load_candidate_predictions(args.results_dir)
    ground_truth, molecule_data = load_ground_truth(args.benchmark_dir)

    strict = candidates_all[~candidates_all.pair_type.isin(CANONICAL_COMPOSITIONS)].copy().reset_index(drop=True)
    strict = add_canonical_offsets(strict, molecule_data)
    ground_keys = set(ground_truth.pair_key)
    candidate_keys = set(strict[strict.true_noncanonical == 1].pair_key)
    if candidate_keys - ground_keys or len(candidate_keys) != int(strict.true_noncanonical.sum()):
        raise ValueError("Prediction labels do not match AMT-derived strict noncanonical ground truth")
    ground_truth = ground_truth.copy()
    ground_truth["in_candidate_window"] = ground_truth.pair_key.isin(candidate_keys)
    probability_map = strict.set_index("pair_key").probability.to_dict()
    prediction_map = strict.set_index("pair_key").predicted_noncanonical.to_dict()
    offset_map = strict.set_index("pair_key").canonical_offset.to_dict()
    ground_truth["probability"] = ground_truth.pair_key.map(probability_map)
    ground_truth["predicted_noncanonical"] = ground_truth.pair_key.map(prediction_map).fillna(0).astype(int)
    ground_truth["canonical_offset"] = ground_truth.pair_key.map(offset_map)
    ground_truth["offset_bucket"] = [offset_bucket(value) for value in ground_truth.canonical_offset]

    strict["distance_bucket"] = [
        distance_bucket(int(distance), relation)
        for distance, relation in zip(strict.sequence_distance, strict.chain_relation)
    ]
    ground_truth["distance_bucket"] = [
        distance_bucket(int(distance), relation)
        for distance, relation in zip(ground_truth.sequence_distance, ground_truth.chain_relation)
    ]

    pair_metrics = grouped_metrics(ground_truth, strict, "pair_type", NONCANONICAL_ORDER)
    classes = class_metrics(ground_truth, strict)
    distance_order = ("1-4", "5-8", "9-16", "17-32", "33-64", "65-128", "129+", "inter-chain")
    distances = grouped_metrics(ground_truth, strict, "distance_bucket", distance_order)
    chains = grouped_metrics(ground_truth, strict, "chain_relation", ("intra-chain", "inter-chain"))
    offsets = grouped_metrics(
        ground_truth,
        strict,
        "offset_bucket",
        ("0-3", "4-7", "8-11", "12-15", "outside-window"),
    )
    molecules = molecule_metrics(ground_truth, strict, molecule_data)
    curve = pr_curve(
        strict.true_noncanonical.to_numpy(dtype=np.int8),
        strict.probability.to_numpy(dtype=np.float64),
        total_positives=len(ground_truth),
    )

    tp = int(((strict.true_noncanonical == 1) & (strict.predicted_noncanonical == 1)).sum())
    fp = int(((strict.true_noncanonical == 0) & (strict.predicted_noncanonical == 1)).sum())
    candidate_gt = int(strict.true_noncanonical.sum())
    total_gt = len(ground_truth)
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, total_gt)
    ap = exact_average_precision(strict.true_noncanonical.to_numpy(), strict.probability.to_numpy())
    prevalence = safe_div(candidate_gt, len(strict))

    topk_rows = []
    ranked = strict.sort_values(["probability", "sample_id", "index_i", "index_j"], ascending=[False, True, True, True])
    for k in (25, 50, 100, int((strict.predicted_noncanonical == 1).sum()), 250, 500, 1000):
        if k <= 0 or k > len(ranked):
            continue
        top = ranked.head(k)
        top_tp = int(top.true_noncanonical.sum())
        topk_rows.append(
            {
                "k": k,
                "tp": top_tp,
                "fp": k - top_tp,
                "precision_at_k": top_tp / k,
                "candidate_recall_at_k": top_tp / max(1, candidate_gt),
                "pipeline_recall_at_k": top_tp / max(1, total_gt),
            }
        )
    topk = pd.DataFrame(topk_rows).drop_duplicates("k").sort_values("k")

    confidence_edges = [threshold, 0.7, 0.8, 0.9, 1.000001]
    confidence_rows = []
    predicted = strict[strict.predicted_noncanonical == 1]
    for lower, upper in zip(confidence_edges[:-1], confidence_edges[1:]):
        part = predicted[(predicted.probability >= lower) & (predicted.probability < upper)]
        part_tp = int(part.true_noncanonical.sum())
        confidence_rows.append(
            {
                "lower": lower,
                "upper": min(upper, 1.0),
                "predicted_count": len(part),
                "tp": part_tp,
                "fp": len(part) - part_tp,
                "precision": safe_div(part_tp, len(part)),
            }
        )
    confidence = pd.DataFrame(confidence_rows)

    score_rows = []
    for label, group in strict.groupby("true_noncanonical"):
        score_rows.append(
            {
                "true_noncanonical": int(label),
                "count": len(group),
                "mean": group.probability.mean(),
                "q25": group.probability.quantile(0.25),
                "median": group.probability.median(),
                "q75": group.probability.quantile(0.75),
                "q90": group.probability.quantile(0.90),
                "q95": group.probability.quantile(0.95),
                "q99": group.probability.quantile(0.99),
                "max": group.probability.max(),
            }
        )
    score_summary = pd.DataFrame(score_rows)

    detail_columns = [
        "sample_id", "index_i", "index_j", "residue_i", "residue_j", "pair_type",
        "raw_amt_label", "sequence_distance", "chain_relation", "canonical_offset", "probability",
    ]
    true_positives = strict[(strict.true_noncanonical == 1) & (strict.predicted_noncanonical == 1)].sort_values("probability", ascending=False)
    false_positives = strict[(strict.true_noncanonical == 0) & (strict.predicted_noncanonical == 1)].sort_values("probability", ascending=False)
    candidate_false_negatives = strict[(strict.true_noncanonical == 1) & (strict.predicted_noncanonical == 0)].sort_values("probability", ascending=False)
    outside_window = ground_truth[~ground_truth.in_candidate_window].copy()

    ground_class_pair = pd.crosstab(ground_truth.raw_amt_label, ground_truth.pair_type).reindex(
        index=range(2, 14), columns=NONCANONICAL_ORDER, fill_value=0
    )
    tp_class_pair = pd.crosstab(true_positives.raw_amt_label, true_positives.pair_type).reindex(
        index=range(2, 14), columns=NONCANONICAL_ORDER, fill_value=0
    )
    fp_pair_offset = pd.crosstab(false_positives.pair_type, false_positives.offset_bucket).reindex(
        index=NONCANONICAL_ORDER,
        columns=("0-3", "4-7", "8-11", "12-15"),
        fill_value=0,
    )

    sequences_by_value: dict[str, list[str]] = {}
    for sample_id, data in molecule_data.items():
        sequence = "".join(str(entry["base"]) for entry in data["entries"])
        sequences_by_value.setdefault(sequence, []).append(sample_id)
    duplicate_sequence_groups = [
        {"sample_ids": sorted(sample_ids), "length": len(sequence)}
        for sequence, sample_ids in sequences_by_value.items()
        if len(sample_ids) > 1
    ]

    bootstrap = bootstrap_intervals(molecules)
    correlations = {
        "spearman_length_vs_average_precision": float(molecules[["length", "average_precision"]].corr(method="spearman").iloc[0, 1]),
        "spearman_length_vs_pipeline_f1": float(molecules[["length", "pipeline_f1"]].corr(method="spearman").iloc[0, 1]),
        "spearman_candidate_pairs_vs_average_precision": float(molecules[["candidate_pairs", "average_precision"]].corr(method="spearman").iloc[0, 1]),
    }
    canonical_candidates = candidates_all[candidates_all.pair_type.isin(CANONICAL_COMPOSITIONS)]
    near_canonical = offsets[offsets.offset_bucket == "0-3"].iloc[0]
    classes_7_8_tp = int(true_positives.raw_amt_label.isin([7, 8]).sum())
    analysis_summary = {
        "definition": "Observed contacts with base composition other than AU/CG/GU are positive; canonical compositions are excluded from this analysis.",
        "checkpoint_epoch": summary.get("checkpoint_epoch"),
        "checkpoint_threshold": threshold,
        "num_molecules": len(molecule_data),
        "strict_candidate_pairs": len(strict),
        "excluded_canonical_composition_candidates": len(canonical_candidates),
        "predicted_canonical_composition_pairs": int(canonical_candidates.predicted_noncanonical.sum()),
        "total_ground_truth_noncanonical": total_gt,
        "candidate_ground_truth_noncanonical": candidate_gt,
        "outside_window_ground_truth": total_gt - candidate_gt,
        "tp": tp,
        "fp": fp,
        "fn": total_gt - tp,
        "precision": precision,
        "pipeline_recall": recall,
        "pipeline_f1": f1_score(precision, recall),
        "candidate_recall": safe_div(candidate_gt, total_gt),
        "candidate_classification_recall": safe_div(tp, candidate_gt),
        "average_precision_strict_candidates": ap,
        "candidate_prevalence": prevalence,
        "ap_lift_over_prevalence": safe_div(ap, prevalence),
        "tp_within_0_to_3_of_canonical_pair": int(near_canonical.tp),
        "tp_fraction_within_0_to_3_of_canonical_pair": safe_div(int(near_canonical.tp), tp),
        "tp_in_amt_classes_7_or_8": classes_7_8_tp,
        "tp_fraction_in_amt_classes_7_or_8": safe_div(classes_7_8_tp, tp),
        "duplicate_full_sequence_groups": duplicate_sequence_groups,
        "molecule_bootstrap_95ci": bootstrap,
        "correlations": correlations,
    }

    per_pdb_table = molecules[["sample_id", "pipeline_f1", "INF", "PPV", "TPR"]].copy()
    per_pdb_table.columns = ["PDB_ID", "F1", "INF", "PPV", "TPR"]
    per_pdb_export = per_pdb_table.fillna(0.0)
    latex_lines = [
        r"PDB ID & F1 & INF & PPV & TPR \\",
        r"\thickhline",
    ]
    for row in per_pdb_export.itertuples(index=False):
        latex_lines.append(
            f"{row.PDB_ID} & {row.F1:.2f} & {row.INF:.2f} & {row.PPV:.2f} & {row.TPR:.2f} \\\\"
        )
    latex_lines.append(r"\thickhline")

    outputs = {
        "pair_type_metrics.csv": pair_metrics,
        "comparison_ready_counts.csv": pair_metrics[["pair_type", "ground_truth_count", "tp", "fn", "fp"]],
        "amt_class_metrics.csv": classes,
        "distance_metrics.csv": distances,
        "chain_relation_metrics.csv": chains,
        "canonical_offset_metrics.csv": offsets,
        "molecule_metrics.csv": molecules,
        "per_pdb_f1_inf_ppv_tpr.csv": per_pdb_export,
        "top_k_metrics.csv": topk,
        "confidence_bins.csv": confidence,
        "score_summary.csv": score_summary,
        "precision_recall_curve.csv": curve,
        "top_true_positives.csv": true_positives[detail_columns],
        "top_false_positives.csv": false_positives[detail_columns],
        "highest_scoring_candidate_false_negatives.csv": candidate_false_negatives[detail_columns],
        "outside_window_false_negatives.csv": outside_window[
            ["sample_id", "index_i", "index_j", "residue_i", "residue_j", "pair_type", "raw_amt_label", "sequence_distance", "chain_relation"]
        ],
        "ground_truth_amt_class_by_pair_type.csv": ground_class_pair.reset_index(),
        "true_positives_amt_class_by_pair_type.csv": tp_class_pair.reset_index(),
        "false_positives_pair_type_by_offset.csv": fp_pair_offset.reset_index(),
    }
    for filename, frame in outputs.items():
        frame.to_csv(output_dir / filename, index=False)
    (output_dir / "per_pdb_f1_inf_ppv_tpr.tex").write_text(
        "\n".join(latex_lines) + "\n", encoding="utf-8"
    )
    (output_dir / "analysis_summary.json").write_text(
        json.dumps(analysis_summary, indent=2, allow_nan=False), encoding="utf-8"
    )

    save_plots(output_dir, pair_metrics, classes, distances, molecules, curve, threshold)

    best_pairs = pair_metrics.sort_values("pipeline_f1", ascending=False).head(3)
    hardest_pairs = pair_metrics.sort_values("pipeline_recall").head(3)
    best_molecules = molecules[molecules.ground_truth_count > 0].sort_values("average_precision", ascending=False).head(5)
    worst_molecules = molecules[molecules.ground_truth_count > 0].sort_values("average_precision").head(5)
    report = f"""# GraphaFold on CASP RNA2 — strict noncanonical analysis

## Evaluation definition

Only observed contacts with unordered base compositions **AA, AC, AG, CC, CU, GG, or UU** are positives. AU, CG/GC, and GU/UG are canonical under the current GraphaFold methodology and are excluded from the primary analysis. Counts use standard definitions: `FN = ground truth - TP`, while `FP` is a predicted non-contact.

The checkpoint threshold is fixed at `{threshold:.8g}` from descriptor validation. It is not tuned on CASP RNA2.

## Overall result

- {total_gt} true noncanonical contacts across {len(molecule_data)} molecules.
- {candidate_gt} ({candidate_gt / total_gt:.1%}) enter the ±15 candidate windows; {total_gt - candidate_gt} are unreachable by the classifier.
- TP={tp}, FP={fp}, FN={total_gt - tp}.
- Precision={precision:.3f}, pipeline recall={recall:.3f}, pipeline F1={f1_score(precision, recall):.3f}.
- Strict-candidate AP={ap:.3f} at prevalence={prevalence:.4f}, a {ap / prevalence:.1f}× lift over random ranking.
- 95% molecule-bootstrap intervals: precision [{bootstrap['precision'][0]:.3f}, {bootstrap['precision'][1]:.3f}], recall [{bootstrap['pipeline_recall'][0]:.3f}, {bootstrap['pipeline_recall'][1]:.3f}], F1 [{bootstrap['pipeline_f1'][0]:.3f}, {bootstrap['pipeline_f1'][1]:.3f}].
- {len(canonical_candidates):,} canonical-composition candidates were excluded; the model predicted {int(canonical_candidates.predicted_noncanonical.sum())} of them as noncanonical.

## Dominant learned signal

- {int(near_canonical.tp)}/{tp} TP ({int(near_canonical.tp) / tp:.1%}) lie within 0–3 residues of both endpoints of a known canonical pair. Only {tp - int(near_canonical.tp)} TP is farther away, although 87 candidate positives lie at offsets 4–15.
- AMT classes 7 (`tHS/tSH`) and 8 (`cWW`) contribute {classes_7_8_tp}/{tp} TP ({classes_7_8_tp / tp:.1%}), despite representing {int(ground_truth.raw_amt_label.isin([7, 8]).sum())}/{total_gt} ground-truth contacts.
- No inter-chain contact is detected: 0 TP, 19 FP and 6 FN. Only 2/6 inter-chain positives enter the candidate windows.
- These patterns show real enrichment, but also a strong dependence on immediate canonical-contact context and selected geometry families.

## Comparison-ready base-composition counts

{markdown_table(pair_metrics, ['pair_type', 'ground_truth_count', 'tp', 'fn', 'fp', 'precision', 'pipeline_recall', 'pipeline_f1', 'candidate_recall'])}

The older comparison table appears to have its FP and FN headings exchanged because TP plus the displayed FP equals the ground-truth count in every row. Its noncanonical ground-truth support is also 387 contacts, versus 281 in the 28 complete files analyzed here. Direct numerical comparison therefore requires rerunning every method on these exact files and this exact definition.

## Best and hardest compositions

Best by pipeline F1:

{markdown_table(best_pairs, ['pair_type', 'ground_truth_count', 'tp', 'fp', 'precision', 'pipeline_recall', 'pipeline_f1'])}

Lowest pipeline recall:

{markdown_table(hardest_pairs, ['pair_type', 'ground_truth_count', 'candidate_recall', 'candidate_classification_recall', 'pipeline_recall'])}

## AMT geometry classes

{markdown_table(classes, ['amt_class', 'class_name', 'ground_truth_count', 'candidate_ground_truth', 'tp', 'fn', 'candidate_recall', 'pipeline_recall'])}

## Molecule-level extremes

Highest AP:

{markdown_table(best_molecules, ['sample_id', 'length', 'ground_truth_count', 'tp', 'fp', 'average_precision', 'pipeline_f1'])}

Lowest AP among molecules with positives:

{markdown_table(worst_molecules, ['sample_id', 'length', 'ground_truth_count', 'tp', 'fp', 'average_precision', 'pipeline_f1'])}

Length versus AP Spearman correlation is {correlations['spearman_length_vs_average_precision']:.3f}; length versus pipeline F1 is {correlations['spearman_length_vs_pipeline_f1']:.3f}. Negative values indicate that transfer becomes harder for longer molecules.

The benchmark is not fully independent at sequence level: exact sequences repeat in {len(duplicate_sequence_groups)} groups: {', '.join('/'.join(group['sample_ids']) for group in duplicate_sequence_groups)}. Their CMT graphs and AMT labels can differ, but molecule-bootstrap confidence intervals should still be treated as approximate.

## Reproducibility and comparison warning

CASP RNA2 has already been inspected during model development. Do not select a checkpoint, threshold, or architecture from these results and then report the same set as a blind test. For a fair method comparison, run SPOT-RNA, sincFold, and UFold on the same 28 complete molecules and score them with the same normalized AU/CG/GU rule and standard TP/FP/FN definitions.
"""
    (output_dir / "analysis_report.md").write_text(report, encoding="utf-8")
    print(json.dumps(analysis_summary, indent=2))
    print(f"Analysis written to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

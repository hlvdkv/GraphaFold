from __future__ import annotations

import csv
import json
import math
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from .data import SampleRecord, _read_square_matrix, load_graph, structure_group
from .labels import is_canonical_base_pair, normalize_base
from .model import GraphaFold
from .training import TrainConfig, _autocast, _metrics


EXPECTED_AMT_VALUES = frozenset(range(-1, 14))
EXPECTED_CMT_VALUES = frozenset({-1, 0, 1})


@dataclass(frozen=True)
class IdxEntry:
    index: int
    code: str
    base: str


def _json_value(value: Any) -> Any:
    """Convert numpy values and non-finite floats to portable JSON values."""

    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_value(value), indent=2, sort_keys=True), encoding="utf-8")


def read_idx_entries(path: Path) -> list[IdxEntry]:
    entries: list[IdxEntry] = []
    seen: set[int] = set()
    with path.open(encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            try:
                index_text, code = raw.rstrip().split(",", 1)
                index = int(index_text)
            except (ValueError, IndexError) as exc:
                raise ValueError(f"Malformed IDX row {path}:{line_no}: {raw.rstrip()!r}") from exc
            if index < 1:
                raise ValueError(f"IDX positions must be one-based and positive in {path}:{line_no}")
            if index in seen:
                raise ValueError(f"Duplicate IDX position {index} in {path}:{line_no}")
            seen.add(index)
            token = code.split(".", 1)[1] if "." in code else code
            entries.append(IdxEntry(index=index, code=code, base=normalize_base(token[:1])))
    if not entries:
        raise ValueError(f"Empty IDX file: {path}")
    indices = [entry.index for entry in entries]
    expected = list(range(1, len(entries) + 1))
    if indices != expected:
        raise ValueError(
            f"IDX positions in {path} must be ordered and consecutive 1..N; "
            f"got first={indices[:5]} last={indices[-5:]}"
        )
    return entries


def _validate_matrix(
    path: Path,
    expected_size: int,
    allowed_values: frozenset[int],
) -> np.ndarray:
    matrix = _read_square_matrix(path)
    if matrix.shape != (expected_size, expected_size):
        raise ValueError(
            f"Matrix/IDX dimension mismatch for {path}: matrix={matrix.shape}, IDX={expected_size}"
        )
    if not np.array_equal(matrix, matrix.T):
        mismatches = int(np.count_nonzero(matrix != matrix.T))
        raise ValueError(f"Matrix {path} is not symmetric ({mismatches} directed mismatches)")
    if np.any(np.diag(matrix) != 0):
        raise ValueError(f"Matrix {path} has a non-zero diagonal")
    values = {int(value) for value in np.unique(matrix)}
    unexpected = sorted(values - allowed_values)
    if unexpected:
        raise ValueError(f"Matrix {path} contains unexpected values: {unexpected}")
    return matrix


def _files_by_stem(directory: Path, suffix: str) -> dict[str, Path]:
    if not directory.is_dir():
        return {}
    return {path.stem: path for path in directory.glob(f"*.{suffix}") if path.is_file()}


def _training_overlap(training_root: Path, benchmark_sequences: dict[str, str]) -> dict[str, Any]:
    training_idx = training_root / "idx"
    if not training_idx.is_dir():
        return {"checked": False, "reason": f"Missing directory: {training_idx}"}

    benchmark_by_sequence: dict[str, list[str]] = {}
    for sample_id, sequence in benchmark_sequences.items():
        benchmark_by_sequence.setdefault(sequence, []).append(sample_id)
    benchmark_groups = {structure_group(sample_id) for sample_id in benchmark_sequences}
    pdb_matches: set[str] = set()
    exact_matches: list[dict[str, str]] = []
    files_checked = 0
    for path in training_idx.glob("*.idx"):
        files_checked += 1
        training_id = path.stem
        group = structure_group(training_id)
        if group in benchmark_groups:
            pdb_matches.add(group)
        try:
            sequence = "".join(entry.base for entry in read_idx_entries(path))
        except ValueError:
            # TrainingSet is audited separately and old descriptor IDX files
            # may be permissive. A malformed training file does not invalidate
            # the benchmark itself.
            continue
        for benchmark_id in benchmark_by_sequence.get(sequence, ()):
            exact_matches.append(
                {"benchmark_id": benchmark_id, "training_id": training_id}
            )
    return {
        "checked": True,
        "training_idx_files_checked": files_checked,
        "pdb_id_matches": sorted(pdb_matches),
        "exact_full_sequence_matches": exact_matches,
    }


def audit_benchmark(
    benchmark_dir: str | Path,
    candidate_window: int = 15,
    training_data_root: str | Path | None = None,
) -> tuple[list[SampleRecord], dict[str, Any], list[dict[str, str]]]:
    root = Path(benchmark_dir)
    paths = {
        kind: _files_by_stem(root / kind, kind)
        for kind in ("amt", "cmt", "idx")
    }
    all_ids = sorted(set().union(*(set(items) for items in paths.values())))
    complete_ids = sorted(set(paths["amt"]) & set(paths["cmt"]) & set(paths["idx"]))
    records: list[SampleRecord] = []
    skipped: list[dict[str, str]] = []
    warnings: list[dict[str, Any]] = []
    lengths: dict[str, int] = {}
    chain_counts: dict[str, int] = {}
    sequences: dict[str, str] = {}
    base_counts: Counter[str] = Counter()
    amt_values: Counter[int] = Counter()
    cmt_values: Counter[int] = Counter()
    totals: Counter[str] = Counter()

    for sample_id in sorted(set(all_ids) - set(complete_ids)):
        missing = [kind for kind in ("amt", "cmt", "idx") if sample_id not in paths[kind]]
        skipped.append(
            {
                "sample_id": sample_id,
                "stage": "format",
                "reason": f"Incomplete AMT/CMT/IDX triple; missing: {', '.join(missing)}",
            }
        )

    for sample_id in complete_ids:
        record = SampleRecord(
            sample_id=sample_id,
            split="test",
            group=structure_group(sample_id),
            amt_path=paths["amt"][sample_id],
            cmt_path=paths["cmt"][sample_id],
            idx_path=paths["idx"][sample_id],
        )
        try:
            entries = read_idx_entries(record.idx_path)
            amt = _validate_matrix(record.amt_path, len(entries), EXPECTED_AMT_VALUES)
            cmt = _validate_matrix(record.cmt_path, len(entries), EXPECTED_CMT_VALUES)
            graph, positives, negatives, bases, total_positives = load_graph(
                record, candidate_window=candidate_window
            )
        except Exception as exc:
            skipped.append(
                {"sample_id": sample_id, "stage": "format", "reason": str(exc)}
            )
            continue

        records.append(record)
        sequence = "".join(entry.base for entry in entries)
        sequences[sample_id] = sequence
        lengths[sample_id] = len(entries)
        chain_counts[sample_id] = len(
            {
                entry.code.split(".", 1)[0] if "." in entry.code else ""
                for entry in entries
            }
        )
        base_counts.update(sequence)
        amt_values.update(int(value) for value in amt.ravel())
        cmt_values.update(int(value) for value in cmt.ravel())
        totals["nodes"] += graph.num_nodes()
        totals["candidate_pairs"] += len(positives) + len(negatives)
        totals["candidate_positives"] += len(positives)
        totals["total_positives"] += total_positives
        totals["molecules_without_candidates"] += int(not positives and not negatives)
        totals["molecules_without_noncanonical_contacts"] += int(total_positives == 0)

        unknown_bases = sum(base == "N" for base in bases)
        if unknown_bases:
            warnings.append(
                {"sample_id": sample_id, "kind": "unknown_bases", "count": unknown_bases}
            )
        mismatch_positions = np.argwhere(np.triu((amt == -1) != (cmt == -1), k=1))
        backbone_mismatch = len(mismatch_positions)
        if backbone_mismatch:
            value_pairs = Counter(
                f"amt={int(amt[i, j])},cmt={int(cmt[i, j])}"
                for i, j in mismatch_positions
            )
            warnings.append(
                {
                    "sample_id": sample_id,
                    "kind": "amt_cmt_backbone_mismatch",
                    "undirected_pairs": backbone_mismatch,
                    "value_pairs": dict(sorted(value_pairs.items())),
                }
            )
        rejected_cmt = 0
        for i in range(len(bases)):
            for j in range(i + 1, len(bases)):
                if cmt[i, j] == 1 and not is_canonical_base_pair(bases[i], bases[j]):
                    rejected_cmt += 1
        if rejected_cmt:
            warnings.append(
                {
                    "sample_id": sample_id,
                    "kind": "cmt_one_with_noncanonical_base_identity",
                    "undirected_pairs": rejected_cmt,
                }
            )

    candidate_recall = (
        totals["candidate_positives"] / totals["total_positives"]
        if totals["total_positives"]
        else None
    )
    report: dict[str, Any] = {
        "benchmark_dir": str(root),
        "candidate_window": candidate_window,
        "file_counts": {kind: len(items) for kind, items in paths.items()},
        "num_unique_sample_ids": len(all_ids),
        "num_complete_triples": len(complete_ids),
        "num_valid_molecules": len(records),
        "num_incomplete_or_invalid": len(skipped),
        "length_min": min(lengths.values()) if lengths else None,
        "length_max": max(lengths.values()) if lengths else None,
        "lengths": lengths,
        "chain_counts": chain_counts,
        "num_multichain_molecules": sum(count > 1 for count in chain_counts.values()),
        "base_counts": dict(sorted(base_counts.items())),
        "amt_value_counts": {str(key): value for key, value in sorted(amt_values.items())},
        "cmt_value_counts": {str(key): value for key, value in sorted(cmt_values.items())},
        "candidate_pairs": totals["candidate_pairs"],
        "candidate_positives": totals["candidate_positives"],
        "total_noncanonical_positives": totals["total_positives"],
        "candidate_recall_ceiling": candidate_recall,
        "molecules_without_candidates": totals["molecules_without_candidates"],
        "molecules_without_noncanonical_contacts": totals[
            "molecules_without_noncanonical_contacts"
        ],
        "warnings": warnings,
        "format_failures": skipped,
    }
    if training_data_root is not None:
        report["training_overlap"] = _training_overlap(Path(training_data_root), sequences)
    return records, report, skipped


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "--device auto requires CUDA for GraphaFold/RiNALMo evaluation. "
                "Run this on the A100 node, or pass --device cpu explicitly for a small diagnostic."
            )
        return torch.device("cuda")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return device


def _load_checkpoint_model(
    checkpoint: dict[str, Any],
    checkpoint_path: Path,
    device: torch.device,
) -> GraphaFold:
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise ValueError(f"Unsupported checkpoint structure: {checkpoint_path}")
    saved = checkpoint.get("config") or {}
    defaults = TrainConfig()

    def setting(name: str):
        return saved.get(name, getattr(defaults, name))

    model = GraphaFold(
        hidden_dim=int(setting("hidden_dim")),
        transformer_layers=int(setting("transformer_layers")),
        transformer_heads=int(setting("transformer_heads")),
        gnn_layers=int(setting("gnn_layers")),
        dropout=float(setting("dropout")),
        sequence_backbone=str(setting("sequence_backbone")),
        rinalmo_model=str(setting("rinalmo_model")),
        freeze_rinalmo=bool(setting("freeze_rinalmo")),
        max_sequence_length=int(setting("max_sequence_length")),
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device).eval()
    return model


def _pipeline_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    candidate_positives: int,
    total_positives: int,
) -> dict[str, float]:
    result = _metrics(labels, probabilities, threshold=threshold)
    predictions = probabilities >= threshold
    tp = int(np.logical_and(predictions, labels == 1).sum())
    fp = int(np.logical_and(predictions, labels == 0).sum())
    candidate_fn = int(np.logical_and(~predictions, labels == 1).sum())
    outside_window = max(0, int(total_positives) - int(candidate_positives))
    pipeline_fn = candidate_fn + outside_window
    pipeline_recall = tp / max(1, tp + pipeline_fn)
    pipeline_f1 = (
        2.0 * result["precision"] * pipeline_recall
        / max(1e-12, result["precision"] + pipeline_recall)
    )
    result.update(
        {
            "candidate_recall": (
                candidate_positives / total_positives if total_positives else float("nan")
            ),
            "candidate_positives": float(candidate_positives),
            "total_positives": float(total_positives),
            "candidate_precision": result["precision"],
            "candidate_classification_recall": result["recall"],
            "candidate_f1": result["f1"],
            "pipeline_recall": pipeline_recall,
            "pipeline_f1": pipeline_f1,
            "true_positives": float(tp),
            "false_positives": float(fp),
            "candidate_false_negatives": float(candidate_fn),
            "outside_window_false_negatives": float(outside_window),
            "pipeline_false_negatives": float(pipeline_fn),
        }
    )
    return result


@torch.inference_mode()
def _predict_one(
    model: GraphaFold,
    record: SampleRecord,
    device: torch.device,
    amp: str,
    threshold: float,
    candidate_window: int,
    pair_chunk: int,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, list[tuple[int, int]], list[IdxEntry]]:
    graph, positives, negatives, _, total_positives = load_graph(
        record, candidate_window=candidate_window
    )
    pairs = sorted(positives + negatives)
    positive_set = set(positives)
    labels = np.asarray([int(pair in positive_set) for pair in pairs], dtype=np.float32)
    probabilities = np.empty(len(pairs), dtype=np.float32)

    if pairs:
        graph = graph.to(device)
        pair_tensor = torch.tensor(pairs, dtype=torch.long)
        with _autocast(device, amp):
            states = model.encode(graph)
        for start in range(0, len(pairs), pair_chunk):
            part = pair_tensor[start : start + pair_chunk].to(device, non_blocking=True)
            with _autocast(device, amp):
                logits = model.score_pairs(graph, states, part)
            probabilities[start : start + len(part)] = torch.sigmoid(logits).float().cpu().numpy()

    metrics = _pipeline_metrics(
        labels,
        probabilities,
        threshold,
        candidate_positives=len(positives),
        total_positives=total_positives,
    )
    metrics.update(
        {
            "sample_id": record.sample_id,
            "length": graph.num_nodes(),
            "predicted_noncanonical": int((probabilities >= threshold).sum()),
        }
    )
    return metrics, labels, probabilities, pairs, read_idx_entries(record.idx_path)


def _write_pair_outputs(
    output_dir: Path,
    record: SampleRecord,
    entries: Sequence[IdxEntry],
    pairs: Sequence[tuple[int, int]],
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> None:
    amt = _read_square_matrix(record.amt_path)
    cmt = _read_square_matrix(record.cmt_path)
    columns = [
        "sample_id",
        "index_i",
        "index_j",
        "residue_i",
        "residue_j",
        "base_i",
        "base_j",
        "sequence_distance",
        "raw_amt_label",
        "true_noncanonical",
        "known_canonical_input",
        "probability",
        "predicted_noncanonical",
    ]
    pairwise = output_dir / "pairwise_predictions" / f"{record.sample_id}.csv"
    predicted = output_dir / "noncanonical_predictions" / f"{record.sample_id}.csv"
    pairwise.parent.mkdir(parents=True, exist_ok=True)
    predicted.parent.mkdir(parents=True, exist_ok=True)
    with pairwise.open("w", newline="", encoding="utf-8") as all_handle, predicted.open(
        "w", newline="", encoding="utf-8"
    ) as positive_handle:
        all_writer = csv.writer(all_handle)
        positive_writer = csv.writer(positive_handle)
        all_writer.writerow(columns)
        positive_writer.writerow(columns)
        for pair, label, probability in zip(pairs, labels, probabilities):
            i, j = pair
            is_prediction = bool(probability >= threshold)
            row = [
                record.sample_id,
                i + 1,
                j + 1,
                entries[i].code,
                entries[j].code,
                entries[i].base,
                entries[j].base,
                abs(i - j),
                int(amt[i, j]),
                int(label),
                int(cmt[i, j] == 1 and is_canonical_base_pair(entries[i].base, entries[j].base)),
                f"{float(probability):.8g}",
                int(is_prediction),
            ]
            all_writer.writerow(row)
            if is_prediction:
                positive_writer.writerow(row)


def _finite_mean(rows: Sequence[dict[str, Any]], key: str) -> float:
    values = [float(row[key]) for row in rows if math.isfinite(float(row[key]))]
    return float(np.mean(values)) if values else float("nan")


def run_evaluation(
    checkpoint_path: str | Path,
    benchmark_dir: str | Path,
    output_dir: str | Path,
    device_name: str = "auto",
    amp: str = "auto",
    threshold: float | None = None,
    candidate_window: int | None = None,
    pair_chunk: int | None = None,
    max_files: int | None = None,
    training_data_root: str | Path | None = None,
) -> dict[str, Any]:
    checkpoint_path = Path(checkpoint_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = _resolve_device(device_name)

    # Load metadata first so the benchmark uses the exact candidate window,
    # precision mode and fixed validation threshold selected during training.
    checkpoint_metadata = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint_metadata, dict) or "model" not in checkpoint_metadata:
        raise ValueError(f"Unsupported checkpoint structure: {checkpoint_path}")
    saved_config = checkpoint_metadata.get("config") or {}
    default_config = TrainConfig()
    used_window = int(
        candidate_window
        if candidate_window is not None
        else saved_config.get("candidate_window", default_config.candidate_window)
    )
    used_amp = str(amp if amp != "auto" else saved_config.get("amp", default_config.amp))
    used_pair_chunk = int(
        pair_chunk
        if pair_chunk is not None
        else saved_config.get("evaluation_pair_chunk", default_config.evaluation_pair_chunk)
    )
    used_threshold = float(
        threshold if threshold is not None else checkpoint_metadata.get("threshold", 0.5)
    )
    if not 0.0 <= used_threshold <= 1.0:
        raise ValueError(f"Threshold must lie in [0, 1], got {used_threshold}")
    if used_window < 0:
        raise ValueError(f"Candidate window cannot be negative, got {used_window}")
    if used_pair_chunk < 1:
        raise ValueError(f"Pair chunk must be positive, got {used_pair_chunk}")
    if max_files is not None and max_files < 1:
        raise ValueError(f"max_files must be positive, got {max_files}")

    records, audit, skipped = audit_benchmark(
        benchmark_dir,
        candidate_window=used_window,
        training_data_root=training_data_root,
    )
    write_json(output_dir / "data_audit.json", audit)
    if max_files is not None:
        records = records[:max_files]
    if not records:
        raise ValueError("No valid benchmark molecules were found")

    checkpoint_epoch = checkpoint_metadata.get("epoch")
    model_load_start = time.perf_counter()
    model = _load_checkpoint_model(checkpoint_metadata, checkpoint_path, device)
    model_load_seconds = time.perf_counter() - model_load_start
    del checkpoint_metadata
    per_molecule: list[dict[str, Any]] = []
    all_labels: list[np.ndarray] = []
    all_probabilities: list[np.ndarray] = []
    for record in records:
        try:
            sequence_start = time.perf_counter()
            prediction_start = time.perf_counter()
            metrics, labels, probabilities, pairs, entries = _predict_one(
                model=model,
                record=record,
                device=device,
                amp=used_amp,
                threshold=used_threshold,
                candidate_window=used_window,
                pair_chunk=used_pair_chunk,
            )
            prediction_seconds = time.perf_counter() - prediction_start
            output_start = time.perf_counter()
            _write_pair_outputs(
                output_dir,
                record,
                entries,
                pairs,
                labels,
                probabilities,
                used_threshold,
            )
            output_seconds = time.perf_counter() - output_start
            metrics.update(
                {
                    "prediction_seconds": prediction_seconds,
                    "output_seconds": output_seconds,
                    "total_sequence_seconds": time.perf_counter() - sequence_start,
                }
            )
            per_molecule.append(metrics)
            all_labels.append(labels)
            all_probabilities.append(probabilities)
            print(
                json.dumps(
                    _json_value(
                        {
                            "sample_id": record.sample_id,
                            "length": metrics["length"],
                            "candidate_pairs": metrics["evaluated_pairs"],
                            "average_precision": metrics["average_precision"],
                            "pipeline_f1": metrics["pipeline_f1"],
                            "total_sequence_seconds": metrics["total_sequence_seconds"],
                        }
                    )
                ),
                flush=True,
            )
        except Exception as exc:
            skipped.append(
                {"sample_id": record.sample_id, "stage": "inference", "reason": repr(exc)}
            )
            print(
                json.dumps(
                    {"sample_id": record.sample_id, "status": "skipped", "reason": repr(exc)}
                ),
                flush=True,
            )

    if not per_molecule:
        write_json(output_dir / "skipped.json", skipped)
        raise RuntimeError("Every valid benchmark molecule failed during inference")

    labels = np.concatenate(all_labels) if all_labels else np.array([], dtype=np.float32)
    probabilities = (
        np.concatenate(all_probabilities) if all_probabilities else np.array([], dtype=np.float32)
    )
    candidate_positives = int(sum(row["candidate_positives"] for row in per_molecule))
    total_positives = int(sum(row["total_positives"] for row in per_molecule))
    micro = _pipeline_metrics(
        labels,
        probabilities,
        used_threshold,
        candidate_positives=candidate_positives,
        total_positives=total_positives,
    )
    macro_keys = (
        "average_precision",
        "precision",
        "recall",
        "f1",
        "positive_prevalence",
        "candidate_recall",
        "pipeline_recall",
        "pipeline_f1",
    )
    macro = {key: _finite_mean(per_molecule, key) for key in macro_keys}
    summary: dict[str, Any] = {
        "mode": "evaluate",
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": checkpoint_epoch,
        "benchmark_dir": str(benchmark_dir),
        "output_dir": str(output_dir),
        "device": str(device),
        "amp": used_amp,
        "threshold": used_threshold,
        "threshold_source": "command_line" if threshold is not None else "checkpoint_validation",
        "candidate_window": used_window,
        "num_molecules": len(per_molecule),
        "num_skipped": len(skipped),
        "model_load_seconds": model_load_seconds,
        "total_sequence_seconds": float(
            sum(float(row["total_sequence_seconds"]) for row in per_molecule)
        ),
        **micro,
        "macro_per_molecule": macro,
        "data_audit_file": str(output_dir / "data_audit.json"),
    }
    write_json(output_dir / "summary.json", summary)
    with (output_dir / "per_molecule.jsonl").open("w", encoding="utf-8") as handle:
        for row in per_molecule:
            handle.write(json.dumps(_json_value(row), sort_keys=True) + "\n")
    write_json(output_dir / "skipped.json", skipped)
    return _json_value(summary)

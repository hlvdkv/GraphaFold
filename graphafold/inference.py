"""Ground-truth-free inference from an IDX sequence and a CMT scaffold."""

from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from .data import BASE_TO_ID, RNAGraph, _directed_edges, window_candidates
from .evaluation import (
    EXPECTED_CMT_VALUES,
    IdxEntry,
    _files_by_stem,
    _json_value,
    _load_checkpoint_model,
    _resolve_device,
    _validate_matrix,
    read_idx_entries,
    write_json,
)
from .labels import is_canonical_base_pair
from .training import TrainConfig, _autocast


@dataclass(frozen=True)
class PredictionRecord:
    sample_id: str
    idx_path: Path
    cmt_path: Path


def discover_prediction_records(
    input_dir: str | Path,
) -> tuple[list[PredictionRecord], list[dict[str, str]]]:
    """Find complete ``idx/<id>.idx`` and ``cmt/<id>.cmt`` input pairs."""

    root = Path(input_dir)
    idx = _files_by_stem(root / "idx", "idx")
    cmt = _files_by_stem(root / "cmt", "cmt")
    all_ids = sorted(set(idx) | set(cmt))
    records: list[PredictionRecord] = []
    skipped: list[dict[str, str]] = []
    for sample_id in all_ids:
        missing = [name for name, files in (("idx", idx), ("cmt", cmt)) if sample_id not in files]
        if missing:
            skipped.append(
                {
                    "sample_id": sample_id,
                    "stage": "format",
                    "reason": f"Incomplete IDX/CMT pair; missing: {', '.join(missing)}",
                }
            )
            continue
        records.append(PredictionRecord(sample_id, idx[sample_id], cmt[sample_id]))
    return records, skipped


def _prediction_graph(
    record: PredictionRecord,
    candidate_window: int,
) -> tuple[RNAGraph, list[tuple[int, int]], list[IdxEntry], np.ndarray]:
    entries = read_idx_entries(record.idx_path)
    cmt = _validate_matrix(record.cmt_path, len(entries), EXPECTED_CMT_VALUES)
    bases = [entry.base for entry in entries]
    backbone: set[tuple[int, int]] = set()
    canonical: set[tuple[int, int]] = set()
    for i, j in zip(*np.triu_indices(len(entries), k=1)):
        pair = (int(i), int(j))
        if int(cmt[i, j]) == -1:
            backbone.add(pair)
        elif int(cmt[i, j]) == 1 and is_canonical_base_pair(bases[i], bases[j]):
            canonical.add(pair)

    candidates = sorted(window_candidates(len(entries), canonical, candidate_window) - backbone)
    edges = [(i, j, 0) for i, j in sorted(backbone)]
    edges.extend((i, j, 1) for i, j in sorted(canonical - backbone))
    src, dst, edge_types = _directed_edges(edges)
    graph = RNAGraph(
        base_id=torch.tensor([BASE_TO_ID[base] for base in bases], dtype=torch.long),
        position=torch.arange(len(entries), dtype=torch.long),
        edge_index=torch.tensor([src, dst], dtype=torch.long).reshape(2, -1),
        edge_type=torch.tensor(edge_types, dtype=torch.long),
        node_counts=(len(entries),),
    )
    return graph, candidates, entries, cmt


@torch.inference_mode()
def _predict_record(
    model,
    record: PredictionRecord,
    device: torch.device,
    amp: str,
    threshold: float,
    candidate_window: int,
    pair_chunk: int,
) -> tuple[dict[str, Any], list[tuple[int, int]], np.ndarray, list[IdxEntry], np.ndarray]:
    graph, pairs, entries, cmt = _prediction_graph(record, candidate_window)
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
    row = {
        "sample_id": record.sample_id,
        "length": len(entries),
        "candidate_pairs": len(pairs),
        "predicted_noncanonical": int((probabilities >= threshold).sum()),
    }
    return row, pairs, probabilities, entries, cmt


def _write_prediction_outputs(
    output_dir: Path,
    record: PredictionRecord,
    entries: Sequence[IdxEntry],
    cmt: np.ndarray,
    pairs: Sequence[tuple[int, int]],
    probabilities: np.ndarray,
    threshold: float,
) -> None:
    columns = [
        "sample_id",
        "index_i",
        "index_j",
        "residue_i",
        "residue_j",
        "base_i",
        "base_j",
        "sequence_distance",
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
        for (i, j), probability in zip(pairs, probabilities):
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
                int(cmt[i, j] == 1 and is_canonical_base_pair(entries[i].base, entries[j].base)),
                f"{float(probability):.8g}",
                int(is_prediction),
            ]
            all_writer.writerow(row)
            if is_prediction:
                positive_writer.writerow(row)


def run_prediction(
    checkpoint_path: str | Path,
    input_dir: str | Path,
    output_dir: str | Path,
    device_name: str = "auto",
    amp: str = "auto",
    threshold: float | None = None,
    candidate_window: int | None = None,
    pair_chunk: int | None = None,
    max_files: int | None = None,
) -> dict[str, Any]:
    """Predict noncanonical contacts without requiring an AMT ground truth."""

    checkpoint_path = Path(checkpoint_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = _resolve_device(device_name)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise ValueError(f"Unsupported checkpoint structure: {checkpoint_path}")
    saved = checkpoint.get("config") or {}
    defaults = TrainConfig()
    used_window = int(
        candidate_window
        if candidate_window is not None
        else saved.get("candidate_window", defaults.candidate_window)
    )
    used_amp = str(amp if amp != "auto" else saved.get("amp", defaults.amp))
    used_pair_chunk = int(
        pair_chunk
        if pair_chunk is not None
        else saved.get("evaluation_pair_chunk", defaults.evaluation_pair_chunk)
    )
    used_threshold = float(threshold if threshold is not None else checkpoint.get("threshold", 0.5))
    if not 0.0 <= used_threshold <= 1.0:
        raise ValueError(f"Threshold must lie in [0, 1], got {used_threshold}")
    if used_window < 0:
        raise ValueError(f"Candidate window cannot be negative, got {used_window}")
    if used_pair_chunk < 1:
        raise ValueError(f"Pair chunk must be positive, got {used_pair_chunk}")
    if max_files is not None and max_files < 1:
        raise ValueError(f"max_files must be positive, got {max_files}")

    records, skipped = discover_prediction_records(input_dir)
    if max_files is not None:
        records = records[:max_files]
    if not records:
        write_json(output_dir / "skipped.json", skipped)
        raise ValueError("No complete IDX/CMT input pairs were found")

    load_start = time.perf_counter()
    model = _load_checkpoint_model(checkpoint, checkpoint_path, device)
    model_load_seconds = time.perf_counter() - load_start
    checkpoint_epoch = checkpoint.get("epoch")
    del checkpoint
    per_molecule: list[dict[str, Any]] = []
    for record in records:
        try:
            started = time.perf_counter()
            row, pairs, probabilities, entries, cmt = _predict_record(
                model=model,
                record=record,
                device=device,
                amp=used_amp,
                threshold=used_threshold,
                candidate_window=used_window,
                pair_chunk=used_pair_chunk,
            )
            _write_prediction_outputs(
                output_dir, record, entries, cmt, pairs, probabilities, used_threshold
            )
            row["total_sequence_seconds"] = time.perf_counter() - started
            per_molecule.append(row)
            print(json.dumps(_json_value(row)), flush=True)
        except Exception as exc:
            skipped.append(
                {"sample_id": record.sample_id, "stage": "inference", "reason": repr(exc)}
            )
            print(
                json.dumps({"sample_id": record.sample_id, "status": "skipped", "reason": repr(exc)}),
                flush=True,
            )

    if not per_molecule:
        write_json(output_dir / "skipped.json", skipped)
        raise RuntimeError("Every input molecule failed during inference")
    summary = {
        "mode": "predict",
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": checkpoint_epoch,
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "device": str(device),
        "amp": used_amp,
        "threshold": used_threshold,
        "threshold_source": "command_line" if threshold is not None else "checkpoint_validation",
        "candidate_window": used_window,
        "num_molecules": len(per_molecule),
        "num_skipped": len(skipped),
        "candidate_pairs": sum(int(row["candidate_pairs"]) for row in per_molecule),
        "predicted_noncanonical": sum(
            int(row["predicted_noncanonical"]) for row in per_molecule
        ),
        "model_load_seconds": model_load_seconds,
        "total_sequence_seconds": sum(
            float(row["total_sequence_seconds"]) for row in per_molecule
        ),
    }
    write_json(output_dir / "summary.json", summary)
    with (output_dir / "per_molecule.jsonl").open("w", encoding="utf-8") as handle:
        for row in per_molecule:
            handle.write(json.dumps(_json_value(row), sort_keys=True) + "\n")
    write_json(output_dir / "skipped.json", skipped)
    return _json_value(summary)


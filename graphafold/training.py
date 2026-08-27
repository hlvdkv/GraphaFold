from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .data import (
    GroupBalancedSampler,
    RNAInteractionDataset,
    collate_graphs,
    read_manifest,
)
from .model import GraphaFold


@dataclass
class TrainConfig:
    data_root: str = "TrainingSet"
    split_csv: str = "split.csv"
    output_dir: str = "runs/GraphaFold"
    seed: int = 42
    hidden_dim: int = 384
    transformer_layers: int = 2
    transformer_heads: int = 8
    gnn_layers: int = 8
    dropout: float = 0.12
    candidate_window: int = 15
    # Full ±15 candidate pools contain roughly 225 negatives per positive in
    # train. Sampling 128 exposes the rare deployment prior without making the
    # pair head score every easy negative on every epoch; set_epoch rotates the
    # sampled pool between epochs.
    negative_ratio: int = 128
    batch_size: int = 12
    epochs: int = 80
    learning_rate: float = 2e-4
    rinalmo_learning_rate: float = 2e-5
    weight_decay: float = 1e-2
    warmup_fraction: float = 0.06
    grad_clip: float = 1.0
    gradient_accumulation: int = 2
    patience: int = 12
    workers: int = 8
    train_windows_per_pdb: int = 256
    evaluation_pair_chunk: int = 131_072
    max_sequence_length: int = 1024
    cache_size: int = 128
    amp: str = "bf16"
    sequence_backbone: str = "rinalmo"
    rinalmo_model: str = "multimolecule/rinalmo-mega"
    freeze_rinalmo: bool = True


class AsymmetricFocalLoss(nn.Module):
    def __init__(self, gamma_positive: float = 1.0, gamma_negative: float = 3.0) -> None:
        super().__init__()
        self.gamma_positive = gamma_positive
        self.gamma_negative = gamma_negative

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        base = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probabilities = torch.sigmoid(logits)
        positive_factor = (1.0 - probabilities).pow(self.gamma_positive)
        negative_factor = probabilities.pow(self.gamma_negative)
        factor = torch.where(targets > 0.5, positive_factor, negative_factor)
        return (base * factor).mean()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _autocast(device: torch.device, amp: str):
    enabled = device.type == "cuda" and amp != "off"
    dtype = torch.bfloat16 if amp == "bf16" else torch.float16
    return torch.autocast(device_type=device.type, enabled=enabled, dtype=dtype)


def _classification_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    """Return thresholded metrics without sorting scores or calculating AP."""
    if labels.size == 0:
        return {
            "threshold": float(threshold),
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "evaluated_pairs": 0.0,
            "positive_prevalence": float("nan"),
        }
    predictions = probabilities >= threshold
    positives = labels == 1
    tp = float(np.logical_and(predictions, positives).sum())
    fp = float(np.logical_and(predictions, ~positives).sum())
    fn = float(np.logical_and(~predictions, positives).sum())
    precision = tp / max(1.0, tp + fp)
    recall = tp / max(1.0, tp + fn)
    f1 = 2.0 * precision * recall / max(1e-12, precision + recall)
    return {
        "threshold": float(threshold),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "evaluated_pairs": float(labels.size),
        "positive_prevalence": float(labels.mean()),
    }


def _asymmetric_focal_loss_from_probabilities(
    labels: np.ndarray,
    probabilities: np.ndarray,
    gamma_positive: float = 1.0,
    gamma_negative: float = 3.0,
) -> float:
    """Calculate validation loss on CPU without changing the model execution."""
    if labels.size == 0:
        return float("nan")
    probabilities = np.clip(probabilities.astype(np.float64), 1e-12, 1.0 - 1e-12)
    labels = labels.astype(np.float64)
    base = -(labels * np.log(probabilities) + (1.0 - labels) * np.log1p(-probabilities))
    positive_factor = np.power(1.0 - probabilities, gamma_positive)
    negative_factor = np.power(probabilities, gamma_negative)
    factor = np.where(labels > 0.5, positive_factor, negative_factor)
    return float(np.mean(base * factor))


def _metrics(labels: np.ndarray, probabilities: np.ndarray, threshold: float | None = None) -> dict[str, float]:
    if labels.size == 0:
        return {
            "average_precision": float("nan"),
            "threshold": float(0.5 if threshold is None else threshold),
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "evaluated_pairs": 0.0,
            "positive_prevalence": float("nan"),
        }

    # A benchmark molecule may contain no noncanonical contact. Keep the
    # checkpoint threshold and still account for false positives instead of
    # silently replacing the threshold with 0.5. AP is undefined without a
    # positive example, while the thresholded precision/recall/F1 remain
    # useful and well-defined by the zero-division convention below.
    if np.unique(labels).size < 2:
        fixed_threshold = float(0.5 if threshold is None else threshold)
        predictions = probabilities >= fixed_threshold
        tp = float(np.logical_and(predictions, labels == 1).sum())
        fp = float(np.logical_and(predictions, labels == 0).sum())
        fn = float(np.logical_and(~predictions, labels == 1).sum())
        precision = tp / max(1.0, tp + fp)
        recall = tp / max(1.0, tp + fn)
        f1 = 2.0 * precision * recall / max(1e-12, precision + recall)
        return {
            "average_precision": float("nan") if labels.sum() == 0 else 1.0,
            "threshold": fixed_threshold,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "evaluated_pairs": float(labels.size),
            "positive_prevalence": float(labels.mean()),
        }
    order = np.argsort(-probabilities, kind="stable")
    sorted_labels = labels[order]
    sorted_probabilities = probabilities[order]
    true_positives = np.cumsum(sorted_labels)
    ranks = np.arange(1, len(labels) + 1)
    positive_count = max(1.0, labels.sum())
    end_of_tie = np.empty(len(labels), dtype=bool)
    end_of_tie[-1] = True
    end_of_tie[:-1] = sorted_probabilities[:-1] != sorted_probabilities[1:]
    tie_ends = np.flatnonzero(end_of_tie)
    true_positives_at_threshold = true_positives[tie_ends]
    positives_per_tie = np.diff(
        np.concatenate((np.zeros(1, dtype=true_positives.dtype), true_positives_at_threshold))
    )
    precision_at_threshold = true_positives_at_threshold / ranks[tie_ends]
    average_precision = float((precision_at_threshold * positives_per_tie).sum() / positive_count)

    def scores(candidate: float) -> tuple[float, float, float]:
        predictions = probabilities >= candidate
        tp = float(np.logical_and(predictions, labels == 1).sum())
        fp = float(np.logical_and(predictions, labels == 0).sum())
        fn = float(np.logical_and(~predictions, labels == 1).sum())
        precision = tp / max(1.0, tp + fp)
        recall = tp / max(1.0, tp + fn)
        f1 = 2.0 * precision * recall / max(1e-12, precision + recall)
        return precision, recall, f1

    if threshold is None:
        # For rare positives, the optimum can lie above the 99th percentile.
        # Evaluate every distinct score exactly. Only the final element of a
        # tied-score block is valid because `probability >= threshold` includes
        # the complete block (BF16 produces many such ties).
        f1_curve = 2.0 * true_positives / (ranks + positive_count)
        f1_curve[~end_of_tie] = -1.0
        threshold = float(sorted_probabilities[int(np.argmax(f1_curve))])
    precision, recall, f1 = scores(float(threshold))
    return {
        "average_precision": average_precision,
        "threshold": float(threshold),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "evaluated_pairs": float(labels.size),
        "positive_prevalence": float(labels.mean()),
    }


@torch.no_grad()
def evaluate(
    model: GraphaFold,
    loader: DataLoader,
    device: torch.device,
    amp: str,
    threshold: float | None = None,
    pair_chunk: int = 131_072,
) -> dict[str, float]:
    model.eval()
    all_probabilities: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    candidate_positive_count = 0
    total_positive_count = 0
    for graph, pairs, labels, _, _, coverage in loader:
        candidate_positive_count += int(coverage[:, 0].sum())
        total_positive_count += int(coverage[:, 1].sum())
        if len(pairs) == 0:
            continue
        graph = graph.to(device)
        with _autocast(device, amp):
            states = model.encode(graph)
        probability_chunks: list[torch.Tensor] = []
        for start in range(0, len(pairs), pair_chunk):
            pair_part = pairs[start : start + pair_chunk].to(device, non_blocking=True)
            with _autocast(device, amp):
                logits = model.score_pairs(graph, states, pair_part)
            probability_chunks.append(torch.sigmoid(logits).float().cpu())
        all_probabilities.append(torch.cat(probability_chunks))
        all_labels.append(labels.cpu())
    if not all_labels:
        labels_array = np.array([])
        probabilities_array = np.array([])
    else:
        labels_array = torch.cat(all_labels).numpy()
        probabilities_array = torch.cat(all_probabilities).numpy()
    result = _metrics(labels_array, probabilities_array, threshold)
    result["loss"] = _asymmetric_focal_loss_from_probabilities(
        labels_array, probabilities_array
    )
    result["candidate_recall"] = candidate_positive_count / max(1, total_positive_count)
    result["candidate_positives"] = float(candidate_positive_count)
    result["total_positives"] = float(total_positive_count)
    result["candidate_precision"] = result["precision"]
    result["candidate_classification_recall"] = result["recall"]
    result["candidate_f1"] = result["f1"]
    result["pipeline_recall"] = result["recall"] * result["candidate_recall"]
    result["pipeline_f1"] = (
        2.0 * result["precision"] * result["pipeline_recall"]
        / max(1e-12, result["precision"] + result["pipeline_recall"])
    )
    return result


def _lr_schedule(step: int, total_steps: int, warmup_fraction: float) -> float:
    warmup_steps = max(1, round(total_steps * warmup_fraction))
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * progress))


def train(config: TrainConfig) -> dict[str, float]:
    set_seed(config.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("GraphaFold training requires CUDA; use audit_dataset.py for CPU-only data checks")
    device = torch.device("cuda")
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records = read_manifest(config.data_root, config.split_csv)
    train_records = [record for record in records if record.split == "train"]
    val_records = [record for record in records if record.split == "val"]
    test_records = [record for record in records if record.split == "test"]
    if not train_records or not val_records:
        raise ValueError("The manifest must contain non-empty train and val splits")

    train_dataset = RNAInteractionDataset(
        train_records,
        config.negative_ratio,
        config.seed,
        config.cache_size,
        all_candidates=False,
        candidate_window=config.candidate_window,
    )
    val_dataset = RNAInteractionDataset(
        val_records,
        seed=config.seed + 1,
        cache_size=config.cache_size,
        all_candidates=True,
        candidate_window=config.candidate_window,
    )
    train_sampler = GroupBalancedSampler(train_records, config.train_windows_per_pdb, config.seed)
    loader_args = dict(
        batch_size=config.batch_size,
        num_workers=config.workers,
        collate_fn=collate_graphs,
        pin_memory=True,
        persistent_workers=False,
    )
    train_loader = DataLoader(train_dataset, sampler=train_sampler, **loader_args)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_args)

    model = GraphaFold(
        hidden_dim=config.hidden_dim,
        transformer_layers=config.transformer_layers,
        transformer_heads=config.transformer_heads,
        gnn_layers=config.gnn_layers,
        dropout=config.dropout,
        sequence_backbone=config.sequence_backbone,
        rinalmo_model=config.rinalmo_model,
        freeze_rinalmo=config.freeze_rinalmo,
        max_sequence_length=config.max_sequence_length,
    ).to(device)
    backbone_parameters = []
    if model.rinalmo_backbone is not None:
        backbone_parameters = [parameter for parameter in model.rinalmo_backbone.parameters() if parameter.requires_grad]
    backbone_ids = {id(parameter) for parameter in backbone_parameters}
    task_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) not in backbone_ids
    ]
    parameter_groups = [{"params": task_parameters, "lr": config.learning_rate}]
    if backbone_parameters:
        parameter_groups.append({"params": backbone_parameters, "lr": config.rinalmo_learning_rate})
    optimizer = torch.optim.AdamW(parameter_groups, weight_decay=config.weight_decay)
    updates_per_epoch = math.ceil(len(train_loader) / config.gradient_accumulation)
    total_updates = max(1, updates_per_epoch * config.epochs)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: _lr_schedule(step, total_updates, config.warmup_fraction)
    )
    loss_function = AsymmetricFocalLoss()
    use_scaler = config.amp == "fp16"
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)

    best_ap = -math.inf
    best_threshold = 0.5
    stale_epochs = 0
    history_path = output_dir / "history.jsonl"
    with (output_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(asdict(config), handle, indent=2)

    for epoch in range(config.epochs):
        train_dataset.set_epoch(epoch)
        train_sampler.set_epoch(epoch)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        train_logit_chunks: list[torch.Tensor] = []
        train_label_chunks: list[torch.Tensor] = []
        for batch_index, (graph, pairs, labels, _, _, _) in enumerate(train_loader):
            if len(pairs) == 0:
                continue
            labels_cpu = labels
            graph, pairs, labels = graph.to(device), pairs.to(device), labels.to(device)
            with _autocast(device, config.amp):
                logits = model(graph, pairs)
                loss = loss_function(logits, labels) / config.gradient_accumulation
            scaler.scale(loss).backward()
            should_update = (batch_index + 1) % config.gradient_accumulation == 0 or batch_index + 1 == len(train_loader)
            if should_update:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
            # Keep metric bookkeeping outside the optimization path. These
            # tensors are detached and moved to CPU only after backward/update.
            running_loss += float(loss.detach()) * config.gradient_accumulation
            train_logit_chunks.append(logits.detach().cpu().float())
            train_label_chunks.append(labels_cpu.float())
        validation = evaluate(
            model,
            val_loader,
            device,
            config.amp,
            pair_chunk=config.evaluation_pair_chunk,
        )
        if train_label_chunks:
            train_labels = torch.cat(train_label_chunks).numpy()
            train_probabilities = torch.sigmoid(torch.cat(train_logit_chunks)).numpy()
            train_f1 = _classification_metrics(
                train_labels,
                train_probabilities,
                float(validation["threshold"]),
            )["f1"]
        else:
            train_f1 = 0.0
        row = {
            "epoch": epoch + 1,
            "train_loss": running_loss / max(1, len(train_loader)),
            "train_f1": train_f1,
            "val_loss": validation["loss"],
            "val_f1": validation["f1"],
            "learning_rate": optimizer.param_groups[0]["lr"],
            "val_average_precision": validation["average_precision"],
            "val_threshold": validation["threshold"],
        }
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)

        if validation["average_precision"] > best_ap:
            best_ap = validation["average_precision"]
            best_threshold = validation["threshold"]
            stale_epochs = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "config": asdict(config),
                    "epoch": epoch + 1,
                    "sampled_val_metrics": validation,
                    "threshold": best_threshold,
                    "label_semantics": "AU/CG/GU=canonical; other observed base combinations=noncanonical",
                },
                output_dir / "best.pt",
            )
        else:
            stale_epochs += 1
        if stale_epochs >= config.patience:
            break

    checkpoint = torch.load(output_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    full_val_loader = DataLoader(
        RNAInteractionDataset(
            val_records,
            all_candidates=True,
            cache_size=0,
            candidate_window=config.candidate_window,
        ),
        shuffle=False,
        **loader_args,
    )
    final = evaluate(
        model,
        full_val_loader,
        device,
        config.amp,
        threshold=best_threshold,
        pair_chunk=config.evaluation_pair_chunk,
    )
    results = {f"full_val_{key}": value for key, value in final.items()}

    if test_records:
        test_loader = DataLoader(
            RNAInteractionDataset(
                test_records,
                all_candidates=True,
                cache_size=0,
                candidate_window=config.candidate_window,
            ),
            shuffle=False,
            **loader_args,
        )
        test_metrics = evaluate(
            model,
            test_loader,
            device,
            config.amp,
            threshold=best_threshold,
            pair_chunk=config.evaluation_pair_chunk,
        )
        results.update({f"test_{key}": value for key, value in test_metrics.items()})

    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    return results

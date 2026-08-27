from __future__ import annotations

import csv
import random
import warnings
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from .labels import InteractionKind, classify_interaction, is_canonical_base_pair, normalize_base


BASE_TO_ID = {"A": 0, "C": 1, "G": 2, "U": 3, "N": 4}


@dataclass(frozen=True)
class SampleRecord:
    sample_id: str
    split: str
    group: str
    amt_path: Path
    cmt_path: Path
    idx_path: Path


@dataclass
class GraphExample:
    graph: "RNAGraph"
    pairs: torch.Tensor
    labels: torch.Tensor
    sample_id: str
    group: str
    candidate_positives: int
    total_positives: int


@dataclass
class RNAGraph:
    """Minimal tensor graph; avoids a compiled PyTorch/DGL compatibility lock."""

    base_id: torch.Tensor
    position: torch.Tensor
    edge_index: torch.Tensor
    edge_type: torch.Tensor
    node_counts: tuple[int, ...]

    def num_nodes(self) -> int:
        return int(self.base_id.shape[0])

    def num_edges(self) -> int:
        return int(self.edge_type.shape[0])

    def batch_num_nodes(self) -> torch.Tensor:
        return torch.tensor(self.node_counts, device=self.base_id.device)

    def to(self, device: torch.device | str) -> "RNAGraph":
        return RNAGraph(
            base_id=self.base_id.to(device, non_blocking=True),
            position=self.position.to(device, non_blocking=True),
            edge_index=self.edge_index.to(device, non_blocking=True),
            edge_type=self.edge_type.to(device, non_blocking=True),
            node_counts=self.node_counts,
        )

    def pin_memory(self) -> "RNAGraph":
        return RNAGraph(
            base_id=self.base_id.pin_memory(),
            position=self.position.pin_memory(),
            edge_index=self.edge_index.pin_memory(),
            edge_type=self.edge_type.pin_memory(),
            node_counts=self.node_counts,
        )


def structure_group(sample_id: str) -> str:
    """PDB-level group used to prevent homologous-window leakage."""

    return sample_id.split("_", 1)[0].upper()


def read_manifest(data_root: str | Path, split_csv: str | Path) -> list[SampleRecord]:
    root = Path(data_root)
    records: list[SampleRecord] = []
    seen: set[str] = set()
    missing: list[str] = []
    with Path(split_csv).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or not {"id", "set"}.issubset(reader.fieldnames):
            raise ValueError(f"{split_csv} must have the columns 'id,set'")
        for line_no, row in enumerate(reader, start=2):
            sample_id = Path((row.get("id") or "").strip()).stem
            split = (row.get("set") or "").strip().lower()
            if not sample_id or split not in {"train", "val", "test"}:
                raise ValueError(f"Invalid manifest row {line_no}: {row!r}")
            if sample_id in seen:
                raise ValueError(f"Duplicate sample id in manifest: {sample_id}")
            seen.add(sample_id)
            record = SampleRecord(
                sample_id=sample_id,
                split=split,
                group=structure_group(sample_id),
                amt_path=root / "amt" / f"{sample_id}.amt",
                cmt_path=root / "cmt" / f"{sample_id}.cmt",
                idx_path=root / "idx" / f"{sample_id}.idx",
            )
            if all(path.is_file() for path in (record.amt_path, record.cmt_path, record.idx_path)):
                records.append(record)
            else:
                missing.append(sample_id)
    validate_group_split(records)
    if missing:
        preview = ", ".join(missing[:5])
        warnings.warn(
            f"Skipped {len(missing)} manifest samples with an incomplete AMT/CMT/IDX triple "
            f"(first: {preview})",
            stacklevel=2,
        )
    return records


def validate_group_split(records: Sequence[SampleRecord]) -> None:
    assignments: dict[str, set[str]] = defaultdict(set)
    for record in records:
        assignments[record.group].add(record.split)
    leaked = {group: splits for group, splits in assignments.items() if len(splits) > 1}
    if leaked:
        preview = ", ".join(f"{g}:{sorted(s)}" for g, s in list(sorted(leaked.items()))[:10])
        raise ValueError(f"PDB-level split leakage detected ({len(leaked)} groups): {preview}")


def read_idx(path: Path) -> list[str]:
    indexed: dict[int, str] = {}
    with path.open(encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            try:
                index_text, code = raw.rstrip().split(",", 1)
                index = int(index_text) - 1
            except (ValueError, IndexError) as exc:
                raise ValueError(f"Malformed IDX row {path}:{line_no}: {raw.rstrip()!r}") from exc
            token = code.split(".", 1)[1] if "." in code else code
            indexed[index] = normalize_base(token[:1])
    if not indexed:
        raise ValueError(f"Empty IDX file: {path}")
    bases = ["N"] * (max(indexed) + 1)
    for index, base in indexed.items():
        bases[index] = base
    return bases


def _read_square_matrix(path: Path) -> np.ndarray:
    matrix = np.loadtxt(path, delimiter=",", dtype=np.int16, ndmin=2)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"Expected a square matrix in {path}, got {matrix.shape}")
    return matrix


def _pad_square(matrix: np.ndarray, size: int) -> np.ndarray:
    if matrix.shape[0] == size:
        return matrix
    if matrix.shape[0] > size:
        return matrix[:size, :size]
    return np.pad(matrix, ((0, size - matrix.shape[0]), (0, size - matrix.shape[1])))


def _directed_edges(undirected: Iterable[tuple[int, int, int]]) -> tuple[list[int], list[int], list[int]]:
    src: list[int] = []
    dst: list[int] = []
    edge_types: list[int] = []
    for i, j, edge_type in undirected:
        src.extend((i, j))
        dst.extend((j, i))
        edge_types.extend((edge_type, edge_type))
    return src, dst, edge_types


def window_candidates(
    size: int,
    canonical_pairs: Iterable[tuple[int, int]],
    radius: int,
) -> set[tuple[int, int]]:
    """Pairs connecting ±radius neighborhoods of canonical-pair endpoints."""

    if radius < 0:
        raise ValueError("candidate window radius cannot be negative")
    if radius == 0:
        return {(i, j) for i in range(size) for j in range(i + 1, size)}
    candidates: set[tuple[int, int]] = set()
    for left, right in canonical_pairs:
        left_range = range(max(0, left - radius), min(size, left + radius + 1))
        right_range = range(max(0, right - radius), min(size, right + radius + 1))
        for i in left_range:
            for j in right_range:
                if i == j:
                    continue
                candidates.add((min(i, j), max(i, j)))
    return candidates


def load_graph(
    record: SampleRecord,
    candidate_window: int = 15,
) -> tuple[RNAGraph, list[tuple[int, int]], list[tuple[int, int]], list[str], int]:
    """Load one graph and return (graph, positives, negative_pool, bases).

    Target labels are derived from AMT only. CMT is an input feature and is
    filtered by nucleotide identities, so a mismatched CMT edge cannot leak a
    noncanonical target into the graph topology.
    """

    bases = read_idx(record.idx_path)
    amt = _read_square_matrix(record.amt_path)
    cmt = _read_square_matrix(record.cmt_path)
    size = max(len(bases), amt.shape[0], cmt.shape[0])
    bases.extend(["N"] * (size - len(bases)))
    amt = _pad_square(amt, size)
    cmt = _pad_square(cmt, size)

    backbone: set[tuple[int, int]] = set()
    canonical_input: set[tuple[int, int]] = set()
    all_positives: set[tuple[int, int]] = set()

    for i in range(size):
        for j in range(i + 1, size):
            raw = int(amt[i, j])
            if raw == -1:
                backbone.add((i, j))
                continue
            if cmt[i, j] == 1 and is_canonical_base_pair(bases[i], bases[j]):
                canonical_input.add((i, j))
            kind = classify_interaction(raw, bases[i], bases[j])
            if kind == InteractionKind.NONCANONICAL:
                all_positives.add((i, j))

    candidates = window_candidates(size, canonical_input, candidate_window) - backbone
    positives = sorted(all_positives & candidates)
    # True canonical contacts are useful hard negatives. No-pair combinations
    # teach the model whether an interaction exists, but only inside the same
    # candidate space used at inference.
    negative_pool = sorted(candidates - all_positives)

    edges = [(i, j, 0) for i, j in sorted(backbone)]
    edges.extend((i, j, 1) for i, j in sorted(canonical_input - backbone))
    src, dst, edge_types = _directed_edges(edges)
    edge_index = torch.tensor([src, dst], dtype=torch.long).reshape(2, -1)
    graph = RNAGraph(
        base_id=torch.tensor([BASE_TO_ID[b] for b in bases], dtype=torch.long),
        position=torch.arange(size, dtype=torch.long),
        edge_index=edge_index,
        edge_type=torch.tensor(edge_types, dtype=torch.long),
        node_counts=(size,),
    )
    return graph, positives, negative_pool, bases, len(all_positives)


def _sample_negatives(
    pool: Sequence[tuple[int, int]],
    positives: Sequence[tuple[int, int]],
    bases: Sequence[str],
    count: int,
    rng: random.Random,
) -> list[tuple[int, int]]:
    if not pool or count <= 0:
        return []

    positive_nodes = {node for pair in positives for node in pair}
    hard_composition: list[tuple[int, int]] = []
    local_context: list[tuple[int, int]] = []
    global_pool: list[tuple[int, int]] = []
    for pair in pool:
        i, j = pair
        if not is_canonical_base_pair(bases[i], bases[j]):
            hard_composition.append(pair)
        elif i in positive_nodes or j in positive_nodes or abs(i - j) <= 32:
            local_context.append(pair)
        else:
            global_pool.append(pair)

    # Most negatives retain a noncanonical letter combination. Otherwise the
    # network can solve the sampled task from pair identity alone.
    quotas = (
        (hard_composition, round(count * 0.60)),
        (local_context, round(count * 0.25)),
        (global_pool, count),
    )
    chosen: list[tuple[int, int]] = []
    used: set[tuple[int, int]] = set()
    for candidates, requested in quotas:
        available = [pair for pair in candidates if pair not in used]
        take = min(requested if len(chosen) < count else 0, count - len(chosen), len(available))
        if take:
            sampled = rng.sample(available, take)
            chosen.extend(sampled)
            used.update(sampled)
    if len(chosen) < count:
        remaining = [pair for pair in pool if pair not in used]
        chosen.extend(rng.sample(remaining, min(count - len(chosen), len(remaining))))
    rng.shuffle(chosen)
    return chosen


class RNAInteractionDataset(Dataset[GraphExample]):
    def __init__(
        self,
        records: Sequence[SampleRecord],
        negative_ratio: int = 12,
        seed: int = 42,
        cache_size: int = 128,
        all_candidates: bool = False,
        candidate_window: int = 15,
    ) -> None:
        self.records = list(records)
        self.negative_ratio = negative_ratio
        self.seed = seed
        self.cache_size = max(0, cache_size)
        self.all_candidates = all_candidates
        self.candidate_window = candidate_window
        self.epoch = 0
        self._cache: OrderedDict[
            int,
            tuple[RNAGraph, list[tuple[int, int]], list[tuple[int, int]], list[str], int],
        ] = OrderedDict()

    def __len__(self) -> int:
        return len(self.records)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def _load(self, index: int):
        if index in self._cache:
            value = self._cache.pop(index)
            self._cache[index] = value
            return value
        value = load_graph(self.records[index], candidate_window=self.candidate_window)
        if self.cache_size:
            self._cache[index] = value
            while len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)
        return value

    def __getitem__(self, index: int) -> GraphExample:
        graph, positives, negative_pool, bases, total_positives = self._load(index)
        if self.all_candidates:
            negatives = list(negative_pool)
        else:
            rng = random.Random((self.seed * 1_000_003) + (self.epoch * 97_409) + index)
            count = self.negative_ratio * max(1, len(positives))
            negatives = _sample_negatives(negative_pool, positives, bases, count, rng)
        pairs = positives + negatives
        labels = [1.0] * len(positives) + [0.0] * len(negatives)
        order = list(range(len(pairs)))
        random.Random(self.seed + self.epoch + index).shuffle(order)
        pair_tensor = torch.tensor([pairs[i] for i in order], dtype=torch.long).reshape(-1, 2)
        label_tensor = torch.tensor([labels[i] for i in order], dtype=torch.float32)
        record = self.records[index]
        return GraphExample(
            graph,
            pair_tensor,
            label_tensor,
            record.sample_id,
            record.group,
            len(positives),
            total_positives,
        )


class GroupBalancedSampler(Sampler[int]):
    """Cap windows per PDB per epoch so ribosomes do not dominate training."""

    def __init__(self, records: Sequence[SampleRecord], max_per_group: int, seed: int = 42) -> None:
        self.by_group: dict[str, list[int]] = defaultdict(list)
        for index, record in enumerate(records):
            self.by_group[record.group].append(index)
        self.max_per_group = max_per_group
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self) -> Iterator[int]:
        rng = random.Random(self.seed + self.epoch)
        indices: list[int] = []
        for group in sorted(self.by_group):
            candidates = self.by_group[group]
            indices.extend(rng.sample(candidates, min(self.max_per_group, len(candidates))))
        rng.shuffle(indices)
        return iter(indices)

    def __len__(self) -> int:
        return sum(min(self.max_per_group, len(v)) for v in self.by_group.values())


def collate_graphs(examples: Sequence[GraphExample]):
    graphs = [example.graph for example in examples]
    offsets = np.cumsum([0] + [graph.num_nodes() for graph in graphs[:-1]])
    edge_parts = [graph.edge_index + int(offset) for graph, offset in zip(graphs, offsets)]
    batched = RNAGraph(
        base_id=torch.cat([graph.base_id for graph in graphs]),
        position=torch.cat([graph.position for graph in graphs]),
        edge_index=torch.cat(edge_parts, dim=1),
        edge_type=torch.cat([graph.edge_type for graph in graphs]),
        node_counts=tuple(graph.num_nodes() for graph in graphs),
    )
    pairs = torch.cat([example.pairs + int(offset) for example, offset in zip(examples, offsets)])
    labels = torch.cat([example.labels for example in examples])
    coverage = torch.tensor(
        [[example.candidate_positives, example.total_positives] for example in examples],
        dtype=torch.long,
    )
    return (
        batched,
        pairs,
        labels,
        [example.sample_id for example in examples],
        [example.group for example in examples],
        coverage,
    )


def select_group_balanced(records: Sequence[SampleRecord], max_per_group: int, seed: int) -> list[SampleRecord]:
    grouped: dict[str, list[SampleRecord]] = defaultdict(list)
    for record in records:
        grouped[record.group].append(record)
    rng = random.Random(seed)
    selected: list[SampleRecord] = []
    for group in sorted(grouped):
        candidates = grouped[group]
        selected.extend(rng.sample(candidates, min(max_per_group, len(candidates))))
    return selected

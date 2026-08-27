#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from graphafold.evaluation import audit_benchmark, run_evaluation, write_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a GraphaFold checkpoint on complete AMT/CMT/IDX molecules. "
            "The validation-selected checkpoint threshold is used by default."
        )
    )
    parser.add_argument("--checkpoint", type=Path, help="GraphaFold best.pt checkpoint.")
    parser.add_argument(
        "--benchmark-dir",
        type=Path,
        default=Path("CaspRNA2"),
        help="Directory containing amt/, cmt/, and idx/ (default: CaspRNA2).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/CaspRNA2"),
        help="Directory for metrics and pair predictions.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="auto (CUDA required), cuda, cuda:0, or explicit cpu for a small diagnostic.",
    )
    parser.add_argument(
        "--amp",
        choices=("auto", "off", "fp16", "bf16"),
        default="auto",
        help="Autocast mode; auto reuses the training checkpoint setting.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Optional fixed override. By default use the threshold selected on validation.",
    )
    parser.add_argument(
        "--candidate-window",
        type=int,
        default=None,
        help="Optional override. By default use candidate_window from the checkpoint.",
    )
    parser.add_argument(
        "--pair-chunk",
        type=int,
        default=None,
        help="Candidate pairs scored per GPU chunk (default: checkpoint setting).",
    )
    parser.add_argument("--max-files", type=int, default=None, help="Evaluate only the first N valid molecules.")
    parser.add_argument(
        "--training-data-root",
        type=Path,
        default=None,
        help="Optional TrainingSet root for PDB-ID and exact full-sequence overlap audit.",
    )
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="Validate benchmark files and candidate coverage without loading a model.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.audit_only:
        window = 15 if args.candidate_window is None else args.candidate_window
        _, report, skipped = audit_benchmark(
            args.benchmark_dir,
            candidate_window=window,
            training_data_root=args.training_data_root,
        )
        args.output_dir.mkdir(parents=True, exist_ok=True)
        write_json(args.output_dir / "data_audit.json", report)
        write_json(args.output_dir / "skipped.json", skipped)
        print(json.dumps({
            "mode": "audit-only",
            "data_audit": str(args.output_dir / "data_audit.json"),
            "num_valid_molecules": report["num_valid_molecules"],
            "num_incomplete_or_invalid": report["num_incomplete_or_invalid"],
            "candidate_recall_ceiling": report["candidate_recall_ceiling"],
        }, indent=2))
        return 0

    if args.checkpoint is None:
        raise SystemExit("--checkpoint is required unless --audit-only is used")
    summary = run_evaluation(
        checkpoint_path=args.checkpoint,
        benchmark_dir=args.benchmark_dir,
        output_dir=args.output_dir,
        device_name=args.device,
        amp=args.amp,
        threshold=args.threshold,
        candidate_window=args.candidate_window,
        pair_chunk=args.pair_chunk,
        max_files=args.max_files,
        training_data_root=args.training_data_root,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

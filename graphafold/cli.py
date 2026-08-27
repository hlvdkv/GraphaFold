"""Command-line interface for prediction, evaluation, audits and training."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from . import __version__
from .training import TrainConfig


def _ablation_defaults() -> TrainConfig:
    return replace(
        TrainConfig(),
        output_dir="runs/GraphaFold-ablation",
        sequence_backbone="learned",
    )


def _add_runtime_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--checkpoint", type=Path, required=True, help="GraphaFold .pt checkpoint.")
    parser.add_argument(
        "--device",
        default="auto",
        help="auto (CUDA required), cuda, cuda:0, or explicit cpu for a small diagnostic.",
    )
    parser.add_argument(
        "--amp",
        choices=("auto", "off", "fp16", "bf16"),
        default="auto",
        help="Autocast mode; auto reuses the checkpoint configuration.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Override the validation-selected checkpoint threshold.",
    )
    parser.add_argument(
        "--candidate-window",
        type=int,
        default=None,
        help="Override the checkpoint candidate-window radius.",
    )
    parser.add_argument(
        "--pair-chunk",
        type=int,
        default=None,
        help="Number of candidate pairs scored per device chunk.",
    )
    parser.add_argument("--max-files", type=int, default=None, help="Process only the first N molecules.")


def _add_train_args(parser: argparse.ArgumentParser, defaults: TrainConfig, ablation: bool) -> None:
    ignored = set()
    if ablation:
        ignored = {
            "sequence_backbone",
            "rinalmo_model",
            "rinalmo_learning_rate",
            "freeze_rinalmo",
        }
    for name in TrainConfig.__dataclass_fields__:
        if name in ignored:
            continue
        default = getattr(defaults, name)
        option = "--" + name.replace("_", "-")
        if isinstance(default, bool):
            parser.add_argument(option, action=argparse.BooleanOptionalAction, default=default)
        else:
            parser.add_argument(option, type=type(default), default=default)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="graphafold",
        description="Predict noncanonical RNA interactions with a sequence-conditioned GNN.",
    )
    parser.add_argument("--version", action="version", version=f"GraphaFold {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    predict = commands.add_parser(
        "predict",
        help="Predict contacts from idx/ and cmt/ without ground-truth AMT matrices.",
    )
    _add_runtime_args(predict)
    predict.add_argument("--input-dir", type=Path, required=True, help="Directory containing idx/ and cmt/.")
    predict.add_argument("--output-dir", type=Path, default=Path("outputs/predictions"))

    evaluate = commands.add_parser(
        "evaluate",
        help="Evaluate a checkpoint on complete amt/, cmt/ and idx/ triples.",
    )
    evaluate.add_argument("--checkpoint", type=Path, default=None, help="GraphaFold .pt checkpoint.")
    evaluate.add_argument("--benchmark-dir", type=Path, required=True)
    evaluate.add_argument("--output-dir", type=Path, default=Path("outputs/evaluation"))
    evaluate.add_argument("--device", default="auto")
    evaluate.add_argument("--amp", choices=("auto", "off", "fp16", "bf16"), default="auto")
    evaluate.add_argument("--threshold", type=float, default=None)
    evaluate.add_argument("--candidate-window", type=int, default=None)
    evaluate.add_argument("--pair-chunk", type=int, default=None)
    evaluate.add_argument("--max-files", type=int, default=None)
    evaluate.add_argument("--training-data-root", type=Path, default=None)
    evaluate.add_argument(
        "--audit-only",
        action="store_true",
        help="Validate files and candidate coverage without loading a model.",
    )

    audit = commands.add_parser("audit", help="Audit training labels and candidate coverage on CPU.")
    audit.add_argument("--data-root", type=Path, default=Path("TrainingSet"))
    audit.add_argument("--split-csv", type=Path, default=Path("split.csv"))
    audit.add_argument("--candidate-window", type=int, default=15)
    audit.add_argument("--max-samples", type=int, default=0)
    audit.add_argument("--output", type=Path, default=Path("label-audit.json"))

    train = commands.add_parser("train", help="Train GraphaFold with the RiNALMo backbone on CUDA.")
    _add_train_args(train, TrainConfig(), ablation=False)

    ablation_defaults = _ablation_defaults()
    ablation = commands.add_parser(
        "train-ablation",
        help="Train GraphaFold-ablation with sequence embeddings learned from scratch.",
    )
    _add_train_args(ablation, ablation_defaults, ablation=True)

    commands.add_parser(
        "check-rinalmo",
        help="Verify pinned dependencies and load the pretrained RiNALMo weights.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "predict":
        from .inference import run_prediction

        result = run_prediction(
            checkpoint_path=args.checkpoint,
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            device_name=args.device,
            amp=args.amp,
            threshold=args.threshold,
            candidate_window=args.candidate_window,
            pair_chunk=args.pair_chunk,
            max_files=args.max_files,
        )
    elif args.command == "evaluate":
        from .evaluation import audit_benchmark, run_evaluation, write_json

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
            result = {
                "mode": "audit-only",
                "data_audit": str(args.output_dir / "data_audit.json"),
                "num_valid_molecules": report["num_valid_molecules"],
                "num_incomplete_or_invalid": report["num_incomplete_or_invalid"],
                "candidate_recall_ceiling": report["candidate_recall_ceiling"],
            }
        else:
            if args.checkpoint is None:
                raise SystemExit("--checkpoint is required unless --audit-only is used")
            result = run_evaluation(
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
    elif args.command == "audit":
        from .audit import audit_training_data

        result = audit_training_data(
            data_root=args.data_root,
            split_csv=args.split_csv,
            candidate_window=args.candidate_window,
            max_samples=args.max_samples,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    elif args.command in {"train", "train-ablation"}:
        from .training import train

        values = vars(args).copy()
        values.pop("command")
        if args.command == "train-ablation":
            ablation_defaults = _ablation_defaults()
            config = replace(
                ablation_defaults,
                **values,
                sequence_backbone="learned",
            )
        else:
            config = TrainConfig(**values)
        result = train(config)
    elif args.command == "check-rinalmo":
        from .diagnostics import check_rinalmo

        result = check_rinalmo()
    else:
        raise AssertionError(f"Unhandled command: {args.command}")

    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

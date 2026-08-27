#!/usr/bin/env python3
"""Train the GraphaFold ablation with sequence embeddings learned from scratch."""

from __future__ import annotations

import argparse
from dataclasses import replace

from graphafold.training import TrainConfig, train


def parse_args() -> TrainConfig:
    defaults = replace(
        TrainConfig(),
        output_dir="runs/GraphaFold-ablation",
        sequence_backbone="learned",
    )
    parser = argparse.ArgumentParser(
        description=(
            "Train the GraphaFold GNN with SDT-style token and positional "
            "embeddings learned from scratch, without RiNALMo."
        )
    )
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
    values = vars(parser.parse_args())
    return replace(defaults, **values, sequence_backbone="learned")


if __name__ == "__main__":
    metrics = train(parse_args())
    print(metrics)

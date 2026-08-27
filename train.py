#!/usr/bin/env python3
from __future__ import annotations

import argparse

from graphafold.training import TrainConfig, train


def parse_args() -> TrainConfig:
    defaults = TrainConfig()
    parser = argparse.ArgumentParser(description="Train the graph-first GraphaFold model")
    for name, field in TrainConfig.__dataclass_fields__.items():
        default = getattr(defaults, name)
        option = "--" + name.replace("_", "-")
        if isinstance(default, bool):
            parser.add_argument(option, action=argparse.BooleanOptionalAction, default=default)
        else:
            parser.add_argument(option, type=type(default), default=default)
    return TrainConfig(**vars(parser.parse_args()))


if __name__ == "__main__":
    metrics = train(parse_args())
    print(metrics)


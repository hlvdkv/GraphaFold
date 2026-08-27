#!/usr/bin/env python3
"""Backward-compatible wrapper for ``graphafold audit``."""

from graphafold.audit import main


if __name__ == "__main__":
    raise SystemExit(main())

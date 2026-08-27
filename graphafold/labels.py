from __future__ import annotations

from enum import IntEnum


VALID_BASES = frozenset("ACGU")

# GU is intentional. In the supplied split, CMT marks 45,208 GU pairs as
# canonical, so wobble pairs must follow the same normalization as AU and CG.
CANONICAL_BASE_PAIRS = frozenset({"AU", "CG", "GU"})


class InteractionKind(IntEnum):
    """Binary GraphaFold convention, while retaining no-pair/backbone states."""

    BACKBONE = -1
    NONE = 0
    CANONICAL = 1
    NONCANONICAL = 2


def normalize_base(base: str) -> str:
    base = (base or "N").strip().upper()
    if base == "T":
        base = "U"
    return base if base in VALID_BASES else "N"


def base_pair_code(base_i: str, base_j: str) -> str:
    """Return an orientation-independent two-letter base-pair code."""

    return "".join(sorted((normalize_base(base_i), normalize_base(base_j))))


def is_canonical_base_pair(base_i: str, base_j: str) -> bool:
    return base_pair_code(base_i, base_j) in CANONICAL_BASE_PAIRS


def classify_interaction(raw_label: int, base_i: str, base_j: str) -> InteractionKind:
    """Normalize an AMT entry for GraphaFold.

    Any observed AU, CG, or GU interaction is canonical, even when the source
    geometry label is greater than 1. An observed interaction made by every
    other base combination is noncanonical. This is deliberately a binary
    task convention; it does not claim a Leontis--Westhof geometry for source
    entries labelled ``1`` with a noncanonical base combination.
    """

    if raw_label == -1:
        return InteractionKind.BACKBONE
    if raw_label <= 0:
        return InteractionKind.NONE
    if is_canonical_base_pair(base_i, base_j):
        return InteractionKind.CANONICAL
    return InteractionKind.NONCANONICAL


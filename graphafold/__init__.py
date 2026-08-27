"""GraphaFold: graph-first prediction of noncanonical RNA interactions."""

from .labels import CANONICAL_BASE_PAIRS, InteractionKind, classify_interaction

__version__ = "0.1.0"

__all__ = [
    "CANONICAL_BASE_PAIRS",
    "InteractionKind",
    "classify_interaction",
    "__version__",
]

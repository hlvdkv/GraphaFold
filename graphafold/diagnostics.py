"""Environment checks for the optional RiNALMo sequence backbone."""

from __future__ import annotations

from importlib.metadata import version
from typing import Any


EXPECTED_MULTIMOLECULE = "0.2.0"
EXPECTED_TRANSFORMERS = "5.9.0"
DEFAULT_RINALMO_MODEL = "multimolecule/rinalmo-mega"


def check_rinalmo(model_name: str = DEFAULT_RINALMO_MODEL) -> dict[str, Any]:
    """Validate dependency versions and verify pretrained-weight loading."""

    installed = {
        "multimolecule": version("multimolecule"),
        "transformers": version("transformers"),
    }
    expected = {
        "multimolecule": EXPECTED_MULTIMOLECULE,
        "transformers": EXPECTED_TRANSFORMERS,
    }
    mismatches = [
        f"{name}=={installed[name]} (expected {wanted})"
        for name, wanted in expected.items()
        if installed[name] != wanted
    ]
    if mismatches:
        raise RuntimeError("Incompatible RiNALMo environment: " + ", ".join(mismatches))

    from multimolecule import RiNALMoModel, RnaTokenizer

    model, loading_info = RiNALMoModel.from_pretrained(
        model_name,
        output_loading_info=True,
    )
    RnaTokenizer.from_pretrained(model_name)
    missing_keys = list(loading_info.get("missing_keys", ()))
    unexpected_keys = list(loading_info.get("unexpected_keys", ()))
    model_key_count = max(1, len(model.state_dict()))
    missing_fraction = len(missing_keys) / model_key_count
    critical_embedding_missing = any(
        key.endswith("embeddings.word_embeddings.weight") for key in missing_keys
    )
    if critical_embedding_missing or missing_fraction > 0.05:
        preview = ", ".join(missing_keys[:8])
        raise RuntimeError(
            "RiNALMo checkpoint was not loaded correctly: "
            f"{len(missing_keys)}/{model_key_count} keys are missing (first: {preview})"
        )
    return {
        **installed,
        "model": model_name,
        "missing_keys": len(missing_keys),
        "unexpected_keys": len(unexpected_keys),
        "status": "RiNALMo weights loaded correctly",
    }


"""Tests for YAML config schema validation (`adhan_slm.core.config_validator`).

Run: PYTHONPATH=src python -m adhan_slm.core.config_validator_tests
"""

from __future__ import annotations

from .config_validator import (
    ConfigValidationError,
    DataConfigSchema,
    ModelConfigSchema,
    TrainingConfigSchema,
)
from .selftest import run_module_tests


def _raises_validation_error(schema, config) -> bool:
    try:
        schema.validate(config)
    except ConfigValidationError:
        return True
    return False


def test_model_config_accepts_a_complete_spec() -> None:
    validated = ModelConfigSchema.validate(
        {
            "model_name": "adhan-tiny",
            "vocab_size": 12000,
            "d_model": 512,
            "num_layers": 8,
            "num_heads": 8,
        }
    )
    assert validated["vocab_size"] == 12000


def test_model_config_rejects_missing_required_field() -> None:
    assert _raises_validation_error(
        ModelConfigSchema, {"model_name": "adhan-tiny", "vocab_size": 12000}
    )


def test_model_config_rejects_wrong_type() -> None:
    """A string vocab_size would only fail much later, inside the embedding shape."""
    assert _raises_validation_error(
        ModelConfigSchema,
        {
            "model_name": "adhan-tiny",
            "vocab_size": "12000",  # should be int
            "d_model": 512,
            "num_layers": 8,
        },
    )


def test_model_config_rejects_out_of_range_value() -> None:
    assert _raises_validation_error(
        ModelConfigSchema,
        {
            "model_name": "adhan-tiny",
            "vocab_size": 10_000_000,  # far past any swaram vocab
            "d_model": 512,
            "num_layers": 8,
        },
    )


def test_training_config_accepts_a_complete_spec() -> None:
    validated = TrainingConfigSchema.validate(
        {"learning_rate": 0.0001, "batch_size": 32, "num_epochs": 3}
    )
    assert validated["batch_size"] == 32


def test_data_config_accepts_a_complete_spec() -> None:
    validated = DataConfigSchema.validate(
        {"corpus_path": "/data/corpus.txt", "seq_length": 1024, "vocab_size": 12000}
    )
    assert validated["seq_length"] == 1024


if __name__ == "__main__":
    run_module_tests(globals(), "core config validation")

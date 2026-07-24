"""Tests for core modules: exceptions, config validation, and metrics."""

import pytest

from .config_validator import (
    ConfigValidationError,
    DataConfigSchema,
    ModelConfigSchema,
    TrainingConfigSchema,
)
from .exceptions import (
    AdhanError,
    ConfigError,
    ExternalDependencyError,
    TokenizerError,
)
from .metrics import LatencyTracker, ThroughputTracker, TrainingMetrics


class TestExceptions:
    """Test custom exception hierarchy."""

    def test_adhan_error_base(self) -> None:
        """Test AdhanError base class."""
        err = AdhanError("Test error", error_code="TEST_001")
        assert "TEST_001" in str(err)
        assert "Test error" in str(err)

    def test_adhan_error_with_context(self) -> None:
        """Test AdhanError with context."""
        err = AdhanError(
            "Tokenization failed",
            error_code="TOKEN_001",
            context={"text": "தமிழ்", "position": 42},
        )
        assert "TOKEN_001" in str(err)
        assert "text=தமிழ்" in str(err)
        assert "position=42" in str(err)

    def test_tokenizer_error_hierarchy(self) -> None:
        """Test TokenizerError is subclass of AdhanError."""
        err = TokenizerError("Vocab mismatch")
        assert isinstance(err, AdhanError)

    def test_external_dependency_error(self) -> None:
        """Test ExternalDependencyError."""
        err = ExternalDependencyError("jax", operation="train")
        assert "jax" in str(err)
        assert "train" in str(err)
        assert "pip install jax" in err.install_cmd


class TestConfigValidation:
    """Test configuration validation schemas."""

    def test_model_config_valid(self) -> None:
        """Test valid model config."""
        config = {
            "model_name": "adhan-tiny",
            "vocab_size": 12000,
            "d_model": 512,
            "num_layers": 8,
            "num_heads": 8,
        }
        validated = ModelConfigSchema.validate(config)
        assert validated["vocab_size"] == 12000

    def test_model_config_missing_required(self) -> None:
        """Test model config with missing required field."""
        config = {"model_name": "adhan-tiny", "vocab_size": 12000}
        with pytest.raises(ConfigValidationError):
            ModelConfigSchema.validate(config)

    def test_model_config_wrong_type(self) -> None:
        """Test model config with wrong field type."""
        config = {
            "model_name": "adhan-tiny",
            "vocab_size": "12000",  # Should be int
            "d_model": 512,
            "num_layers": 8,
        }
        with pytest.raises(ConfigValidationError):
            ModelConfigSchema.validate(config)

    def test_model_config_out_of_range(self) -> None:
        """Test model config with out-of-range value."""
        config = {
            "model_name": "adhan-tiny",
            "vocab_size": 10000000,  # Too large
            "d_model": 512,
            "num_layers": 8,
        }
        with pytest.raises(ConfigValidationError):
            ModelConfigSchema.validate(config)

    def test_training_config_valid(self) -> None:
        """Test valid training config."""
        config = {
            "learning_rate": 0.0001,
            "batch_size": 32,
            "num_epochs": 3,
        }
        validated = TrainingConfigSchema.validate(config)
        assert validated["batch_size"] == 32

    def test_data_config_valid(self) -> None:
        """Test valid data config."""
        config = {
            "corpus_path": "/data/corpus.txt",
            "seq_length": 1024,
            "vocab_size": 12000,
        }
        validated = DataConfigSchema.validate(config)
        assert validated["seq_length"] == 1024


class TestMetrics:
    """Test metrics tracking."""

    def test_throughput_tracker(self) -> None:
        """Test ThroughputTracker."""
        tracker = ThroughputTracker(window_size=10)
        # Add some data points
        for _ in range(5):
            tracker.update(num_tokens=1024, num_examples=32)

        tps = tracker.tokens_per_second()
        assert tps is not None and tps > 0

    def test_latency_tracker(self) -> None:
        """Test LatencyTracker."""
        tracker = LatencyTracker()
        latencies = [0.001, 0.002, 0.0015, 0.0025, 0.002]

        for lat in latencies:
            tracker.update(lat)

        assert tracker.mean() is not None
        assert tracker.min() is not None
        assert tracker.max() is not None
        assert tracker.p95() is not None
        assert tracker.p99() is not None

    def test_training_metrics(self) -> None:
        """Test TrainingMetrics aggregation."""
        metrics = TrainingMetrics()

        # Add some data
        metrics.update_loss(2.5)
        metrics.update_loss(2.3)
        metrics.update_eval_loss(2.4)

        metrics.throughput.update(1024, 32)

        summary = metrics.summary()
        assert "loss_mean" in summary
        assert "loss_current" in summary
        assert summary["loss_current"] == 2.3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

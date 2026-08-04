"""Core utilities for Adhan SLM.

Provides logging, error handling, configuration validation, and metrics tracking.
"""

from .config_validator import (
    ConfigSchema,
    ConfigValidationError,
    DataConfigSchema,
    ModelConfigSchema,
    TrainingConfigSchema,
    load_and_validate_config,
    validate_data_config,
    validate_model_config,
    validate_training_config,
)
from .exceptions import (
    AdhanError,
    CheckpointError,
    ConfigError,
    CorpusError,
    DataError,
    EvaluationError,
    ExternalDependencyError,
    InferenceError,
    ModelError,
    PackingError,
    ShardError,
    TokenizationError,
    TokenizerError,
    TrainingError,
    VocabularyError,
)
from .logging import (
    ColoredFormatter,
    StructuredFormatter,
    configure_root_logger,
    get_logger,
    log_with_context,
)
from .metrics import (
    LatencyTracker,
    ResourceMonitor,
    ThroughputTracker,
    TrainingMetrics,
)
from .selftest import collect_module_tests, run_module_tests, skip_unless

__all__ = [
    # Logging
    "get_logger",
    "configure_root_logger",
    "log_with_context",
    "StructuredFormatter",
    "ColoredFormatter",
    # Exceptions
    "AdhanError",
    "TokenizerError",
    "TokenizationError",
    "VocabularyError",
    "DataError",
    "CorpusError",
    "PackingError",
    "ShardError",
    "ConfigError",
    "ConfigValidationError",
    "TrainingError",
    "CheckpointError",
    "ModelError",
    "InferenceError",
    "EvaluationError",
    "ExternalDependencyError",
    # Config validation
    "ConfigSchema",
    "ModelConfigSchema",
    "TrainingConfigSchema",
    "DataConfigSchema",
    "load_and_validate_config",
    "validate_model_config",
    "validate_training_config",
    "validate_data_config",
    # Metrics
    "ThroughputTracker",
    "LatencyTracker",
    "ResourceMonitor",
    "TrainingMetrics",
    # Test helpers (shared by the *_tests.py modules)
    "run_module_tests",
    "collect_module_tests",
    "skip_unless",
]

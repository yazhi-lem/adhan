"""Configuration validation for Adhan SLM.

Validates YAML configs and model parameters at load time.
"""

from typing import Any, Dict, Optional, Set, Type, Union

import yaml

from .exceptions import ConfigValidationError


class ConfigSchema:
    """Base schema validator for configuration."""

    # Override in subclasses
    REQUIRED_FIELDS: Set[str] = set()
    OPTIONAL_FIELDS: Dict[str, Any] = {}
    FIELD_TYPES: Dict[str, Type] = {}
    FIELD_RANGES: Dict[str, tuple[Union[int, float], Union[int, float]]] = {}

    @classmethod
    def validate(cls, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate configuration dictionary.

        Args:
            config: Configuration dictionary

        Returns:
            Validated configuration

        Raises:
            ConfigValidationError: If validation fails
        """
        # Check required fields
        for field in cls.REQUIRED_FIELDS:
            if field not in config:
                raise ConfigValidationError(
                    f"Missing required field '{field}'",
                    invalid_field=field,
                )

        # Check field types
        for field, value in config.items():
            if field in cls.FIELD_TYPES:
                expected_type = cls.FIELD_TYPES[field]
                if not isinstance(value, expected_type):
                    raise ConfigValidationError(
                        f"Field '{field}' has wrong type",
                        invalid_field=field,
                        expected=expected_type.__name__,
                        got=type(value).__name__,
                    )

            # Check ranges for numeric fields
            if field in cls.FIELD_RANGES:
                min_val, max_val = cls.FIELD_RANGES[field]
                if not (min_val <= value <= max_val):
                    raise ConfigValidationError(
                        f"Field '{field}' out of range [{min_val}, {max_val}]",
                        invalid_field=field,
                        expected=f"[{min_val}, {max_val}]",
                        got=value,
                    )

        return config


class ModelConfigSchema(ConfigSchema):
    """Schema for model configuration."""

    REQUIRED_FIELDS = {"model_name", "vocab_size", "d_model", "num_layers"}

    FIELD_TYPES = {
        "model_name": str,
        "vocab_size": int,
        "d_model": int,
        "num_layers": int,
        "num_heads": int,
        "ffn_dim": int,
        "context_length": int,
        "dropout": float,
        "use_boundary_emb": bool,
    }

    FIELD_RANGES = {
        "vocab_size": (100, 100000),
        "d_model": (64, 2048),
        "num_layers": (1, 32),
        "num_heads": (1, 32),
        "ffn_dim": (128, 8192),
        "context_length": (64, 8192),
        "dropout": (0.0, 0.5),
    }


class TrainingConfigSchema(ConfigSchema):
    """Schema for training configuration."""

    REQUIRED_FIELDS = {"learning_rate", "batch_size", "num_epochs"}

    FIELD_TYPES = {
        "learning_rate": (int, float),
        "batch_size": int,
        "num_epochs": int,
        "warmup_steps": int,
        "eval_every": int,
        "checkpoint_every": int,
        "keep_checkpoints": int,
        "max_grad_norm": (int, float),
        "weight_decay": (int, float),
    }

    FIELD_RANGES = {
        "learning_rate": (1e-6, 0.1),
        "batch_size": (1, 2048),
        "num_epochs": (1, 1000),
        "warmup_steps": (0, 100000),
        "eval_every": (1, 100000),
        "checkpoint_every": (1, 100000),
        "keep_checkpoints": (1, 100),
        "max_grad_norm": (0.1, 10.0),
        "weight_decay": (0.0, 0.1),
    }


class DataConfigSchema(ConfigSchema):
    """Schema for data configuration."""

    REQUIRED_FIELDS = {"corpus_path", "seq_length", "vocab_size"}

    FIELD_TYPES = {
        "corpus_path": str,
        "seq_length": int,
        "vocab_size": int,
        "num_workers": int,
        "prefetch_size": int,
    }

    FIELD_RANGES = {
        "seq_length": (64, 8192),
        "vocab_size": (100, 100000),
        "num_workers": (0, 32),
        "prefetch_size": (1, 1024),
    }


def load_and_validate_config(
    config_path: str,
    schema: Type[ConfigSchema] = ConfigSchema,
) -> Dict[str, Any]:
    """Load YAML config and validate against schema.

    Args:
        config_path: Path to YAML config file
        schema: ConfigSchema subclass to use for validation

    Returns:
        Validated configuration dictionary

    Raises:
        ConfigValidationError: If validation fails
        FileNotFoundError: If config file not found
    """
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError as e:
        raise ConfigValidationError(
            f"Config file not found: {config_path}",
            invalid_field="config_path",
            got=config_path,
        ) from e
    except yaml.YAMLError as e:
        raise ConfigValidationError(
            f"Invalid YAML in config file: {e}",
            invalid_field="yaml_syntax",
        ) from e

    if config is None:
        config = {}

    return schema.validate(config)


def validate_model_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Validate model configuration.

    Args:
        config: Model config dictionary

    Returns:
        Validated configuration
    """
    return ModelConfigSchema.validate(config)


def validate_training_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Validate training configuration.

    Args:
        config: Training config dictionary

    Returns:
        Validated configuration
    """
    return TrainingConfigSchema.validate(config)


def validate_data_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Validate data configuration.

    Args:
        config: Data config dictionary

    Returns:
        Validated configuration
    """
    return DataConfigSchema.validate(config)

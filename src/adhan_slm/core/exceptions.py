"""Custom exception hierarchy for Adhan SLM.

Provides structured error handling with context and recovery strategies.
"""

from typing import Any, Optional


class AdhanError(Exception):
    """Base exception for all Adhan errors."""

    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        context: Optional[dict[str, Any]] = None,
    ) -> None:
        """Initialize AdhanError.

        Args:
            message: Error message
            error_code: Machine-readable error code (e.g., "TOKEN_001")
            context: Additional context (e.g., {"file": "corpus.txt", "line": 42})
        """
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.context = context or {}
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        """Format error message with code and context."""
        msg = f"[{self.error_code}] {self.message}"
        if self.context:
            context_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
            msg += f" ({context_str})"
        return msg


class TokenizerError(AdhanError):
    """Tokenizer-related errors."""

    pass


class TokenizationError(TokenizerError):
    """Raised when tokenization fails."""

    pass


class VocabularyError(TokenizerError):
    """Raised when vocabulary is invalid or missing."""

    pass


class DataError(AdhanError):
    """Data pipeline-related errors."""

    pass


class CorpusError(DataError):
    """Raised when corpus loading/processing fails."""

    pass


class PackingError(DataError):
    """Raised when data packing fails."""

    pass


class ShardError(DataError):
    """Raised when shard reading/writing fails."""

    pass


class ConfigError(AdhanError):
    """Configuration-related errors."""

    pass


class ConfigValidationError(ConfigError):
    """Raised when YAML/config validation fails."""

    def __init__(
        self,
        message: str,
        invalid_field: Optional[str] = None,
        expected: Optional[Any] = None,
        got: Optional[Any] = None,
    ) -> None:
        """Initialize ConfigValidationError.

        Args:
            message: Error message
            invalid_field: Field that failed validation (e.g., "vocab_size")
            expected: Expected type/value
            got: Actual type/value
        """
        self.invalid_field = invalid_field
        self.expected = expected
        self.got = got

        context = {}
        if invalid_field:
            context["field"] = invalid_field
        if expected is not None:
            context["expected"] = str(expected)
        if got is not None:
            context["got"] = str(got)

        super().__init__(message, error_code="CONFIG_VALIDATION", context=context)


class TrainingError(AdhanError):
    """Training-related errors."""

    pass


class CheckpointError(TrainingError):
    """Raised when checkpoint save/load fails."""

    pass


class ModelError(AdhanError):
    """Model-related errors."""

    pass


class InferenceError(ModelError):
    """Raised when inference fails."""

    pass


class EvaluationError(AdhanError):
    """Evaluation-related errors."""

    pass


class ExternalDependencyError(AdhanError):
    """Raised when required external dependency is missing."""

    def __init__(
        self,
        package: str,
        operation: str = "import",
        install_cmd: Optional[str] = None,
    ) -> None:
        """Initialize ExternalDependencyError.

        Args:
            package: Missing package name (e.g., "jax")
            operation: Operation that failed (e.g., "import", "train")
            install_cmd: Command to install the package
        """
        self.package = package
        self.install_cmd = install_cmd or f"pip install {package}"

        message = f"Required package '{package}' not available for {operation}"
        context = {"package": package, "install": self.install_cmd}

        super().__init__(message, error_code="MISSING_DEPENDENCY", context=context)

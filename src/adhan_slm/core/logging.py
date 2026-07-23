"""Structured logging utilities for Adhan SLM.

Provides a unified logging interface with support for both human-readable
(dev/console) and structured JSON (production) output formats.
"""

import json
import logging
import os
import sys
from datetime import datetime
from typing import Any, Dict, Optional


class StructuredFormatter(logging.Formatter):
    """Formats log records as JSON for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_data: Dict[str, Any] = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        # Add any extra fields
        if hasattr(record, "extra_fields"):
            log_data.update(record.extra_fields)

        return json.dumps(log_data)


class ColoredFormatter(logging.Formatter):
    """Formats log records with ANSI colors for console output."""

    # ANSI color codes
    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        """Format log record with color."""
        level_color = self.COLORS.get(record.levelname, "")
        timestamp = datetime.fromtimestamp(record.created).isoformat()

        # Format: [timestamp] [LEVEL] logger.module:function - message
        log_msg = (
            f"[{timestamp}] {level_color}[{record.levelname}]{self.RESET} "
            f"{record.name}:{record.funcName}:{record.lineno} - {record.getMessage()}"
        )

        if record.exc_info:
            log_msg += f"\n{self.formatException(record.exc_info)}"

        return log_msg


def get_logger(
    name: str,
    level: Optional[str] = None,
    use_json: Optional[bool] = None,
) -> logging.Logger:
    """Get a logger instance with structured or colored formatting.

    Args:
        name: Logger name (typically __name__ of the calling module)
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
               Defaults to env var ADHAN_LOG_LEVEL or INFO.
        use_json: If True, use JSON structured output.
                 Defaults to env var ADHAN_JSON_LOGS or False.

    Returns:
        Configured logger instance.
    """
    # Get configuration from environment or defaults
    if level is None:
        level = os.getenv("ADHAN_LOG_LEVEL", "INFO").upper()
    if use_json is None:
        use_json = os.getenv("ADHAN_JSON_LOGS", "false").lower() == "true"

    # Create or get logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level))

    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()

    # Create console handler
    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(getattr(logging, level))

    # Set formatter based on output format
    if use_json:
        formatter = StructuredFormatter()
    else:
        formatter = ColoredFormatter()

    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # Prevent propagation to root logger
    logger.propagate = False

    return logger


def configure_root_logger(
    level: str = "INFO",
    use_json: bool = False,
) -> None:
    """Configure root logger for the entire application.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        use_json: If True, use JSON structured output
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))

    # Remove existing handlers
    root_logger.handlers.clear()

    # Create console handler
    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(getattr(logging, level.upper()))

    # Set formatter
    if use_json:
        formatter = StructuredFormatter()
    else:
        formatter = ColoredFormatter()

    handler.setFormatter(formatter)
    root_logger.addHandler(handler)


# Convenience function for adding contextual fields to logs
def log_with_context(
    logger: logging.Logger,
    level: str,
    message: str,
    **context: Any,
) -> None:
    """Log a message with additional context fields.

    Args:
        logger: Logger instance
        level: Log level as string (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        message: Log message
        **context: Additional fields to include in the log record
    """
    record = logger.makeRecord(
        logger.name,
        getattr(logging, level.upper()),
        "unknown",
        0,
        message,
        (),
        None,
    )
    record.extra_fields = context
    logger.handle(record)

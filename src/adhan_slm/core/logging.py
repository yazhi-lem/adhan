"""Structured logging utilities for Adhan SLM.

Provides a unified logging interface with support for both human-readable
(dev/console) and structured JSON (production) output formats.

Production features:
  * RotatingFileHandler for persistent logs (auto-rollover at 10 MB, keep 5 backups)
  * Structured JSON output for Cloud Logging ingestion
  * Correlation ID support for distributed request tracing
  * ADHAN_LOG_LEVEL / ADHAN_JSON_LOGS / ADHAN_LOG_DIR environment overrides
"""

import json
import logging
import os
import sys
import threading
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
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

        # Add correlation ID if present
        if hasattr(record, "correlation_id"):
            log_data["correlation_id"] = record.correlation_id

        return json.dumps(log_data, ensure_ascii=False)


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


# Thread-local correlation ID storage
_correlation_ctx = threading.local()


def set_correlation_id(cid: str) -> None:
    """Set the correlation ID for the current thread / async context."""
    _correlation_ctx.cid = cid


def get_correlation_id() -> Optional[str]:
    """Get the correlation ID for the current thread, if any."""
    return getattr(_correlation_ctx, "cid", None)


def clear_correlation_id() -> None:
    """Clear the correlation ID for the current thread."""
    if hasattr(_correlation_ctx, "cid"):
        delattr(_correlation_ctx, "cid")


class CorrelationIdFilter(logging.Filter):
    """Inject correlation_id into log records from thread-local storage."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.correlation_id = getattr(_correlation_ctx, "cid", None)
        return True


def get_logger(
    name: str,
    level: Optional[str] = None,
    use_json: Optional[bool] = None,
    log_dir: Optional[str] = None,
    rotation_mb: int = 10,
    backup_count: int = 5,
) -> logging.Logger:
    """Get a logger instance with structured or colored formatting.

    Args:
        name: Logger name (typically __name__ of the calling module)
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
               Defaults to env var ADHAN_LOG_LEVEL or INFO.
        use_json: If True, use JSON structured output.
                 Defaults to env var ADHAN_JSON_LOGS or False.
        log_dir: Directory for rotating log files.
                 Defaults to env var ADHAN_LOG_DIR or None (no file output).
        rotation_mb: Max size per log file before rotation.
        backup_count: Number of backup files to keep.

    Returns:
        Configured logger instance.
    """
    # Get configuration from environment or defaults
    if level is None:
        level = os.getenv("ADHAN_LOG_LEVEL", "INFO").upper()
    if use_json is None:
        use_json = os.getenv("ADHAN_JSON_LOGS", "false").lower() == "true"
    if log_dir is None:
        log_dir = os.getenv("ADHAN_LOG_DIR")

    # Create or get logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level))

    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()

    # Always add correlation ID filter
    logger.addFilter(CorrelationIdFilter())

    # Console handler
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(getattr(logging, level))
    if use_json:
        console_handler.setFormatter(StructuredFormatter())
    else:
        console_handler.setFormatter(ColoredFormatter())
    logger.addHandler(console_handler)

    # Rotating file handler (production persistence)
    if log_dir:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            log_path / "adhan.log",
            maxBytes=rotation_mb * 1024 * 1024,
            backupCount=backup_count,
            encoding="utf-8",
        )
        file_handler.setLevel(getattr(logging, level))
        # Always write structured JSON to the file for log aggregation
        file_handler.setFormatter(StructuredFormatter())
        logger.addHandler(file_handler)

    # Prevent propagation to root logger
    logger.propagate = False

    return logger


def configure_root_logger(
    level: str = "INFO",
    use_json: bool = False,
    log_dir: Optional[str] = None,
) -> None:
    """Configure root logger for the entire application.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        use_json: If True, use JSON structured output
        log_dir: Directory for rotating log files
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))

    # Remove existing handlers
    root_logger.handlers.clear()

    # Console handler
    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(getattr(logging, level.upper()))
    if use_json:
        handler.setFormatter(StructuredFormatter())
    else:
        handler.setFormatter(ColoredFormatter())
    root_logger.addHandler(handler)

    # File handler
    if log_dir:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            log_path / "adhan.log",
            maxBytes=10 * 1024 * 1024,
            backupCount=5,
            encoding="utf-8",
        )
        file_handler.setFormatter(StructuredFormatter())
        root_logger.addHandler(file_handler)


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

"""
Adhan core exceptions and utilities.
"""

import logging
from typing import Optional


class AdhanException(Exception):
    """Base exception for Adhan."""
    pass


class TokenizerException(AdhanException):
    """Tokenizer-related errors."""
    pass


class DataException(AdhanException):
    """Data processing errors."""
    pass


class ModelException(AdhanException):
    """Model training/inference errors."""
    pass


def setup_logger(name: str, level: str = "INFO") -> logging.Logger:
    """Configure logger for Adhan modules."""
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level))
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger

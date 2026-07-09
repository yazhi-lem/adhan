"""Bridges onto external Tamil NLP foundations. See open_tamil_bridge for the
primary one (open-tamil, MIT-licensed, used across tokenizer tests, eval, and
corpus tooling)."""
from .open_tamil_bridge import HAS_OPEN_TAMIL

__all__ = ["HAS_OPEN_TAMIL"]

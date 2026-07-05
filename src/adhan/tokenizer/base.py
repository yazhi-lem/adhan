"""
Abstract tokenizer base class.
"""

from abc import ABC, abstractmethod
from typing import List, Dict
from ..core import TokenizerConfig


class BaseTokenizer(ABC):
    """Base class for all tokenizers."""
    
    def __init__(self, config: TokenizerConfig):
        self.config = config
        self.vocab_size = config.vocab_size
        self.max_length = config.max_length
    
    @abstractmethod
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into tokens."""
        pass
    
    @abstractmethod
    def encode(self, text: str) -> List[int]:
        """Encode text into token IDs."""
        pass
    
    @abstractmethod
    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs back to text."""
        pass
    
    @abstractmethod
    def get_vocab_size(self) -> int:
        """Return vocabulary size."""
        pass

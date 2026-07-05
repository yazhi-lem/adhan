"""
Base model class for Adhan training and inference.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from pathlib import Path
from ..core import ModelConfig, setup_logger

logger = setup_logger(__name__)


class BaseModel(ABC):
    """Abstract base class for all Adhan models."""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.is_loaded = False
    
    @abstractmethod
    def build(self) -> None:
        """Build model architecture."""
        pass
    
    @abstractmethod
    def train(self, train_data: List[str], val_data: Optional[List[str]] = None) -> Dict[str, Any]:
        """Train model on data."""
        pass
    
    @abstractmethod
    def evaluate(self, eval_data: List[str]) -> Dict[str, float]:
        """Evaluate model on data."""
        pass
    
    @abstractmethod
    def save(self, path: str) -> None:
        """Save model checkpoint."""
        pass
    
    @abstractmethod
    def load(self, path: str) -> None:
        """Load model from checkpoint."""
        pass
    
    def get_config(self) -> ModelConfig:
        """Return model config."""
        return self.config
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model_type={self.config.model_type}, device={self.config.device})"

"""
Adhan Core Configuration Module

Centralized configuration for tokenizer, data, training, evaluation.
All parameters in one place; profiles for different environments (RPi5, cluster, local).
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional
import yaml
import logging

logger = logging.getLogger(__name__)


@dataclass
class TokenizerConfig:
    """Tamil tokenizer configuration."""
    vocab_size: int = 32000
    model_type: str = "tamil_agglutinative"
    max_length: int = 512
    morphemes: int = 70  # 70+ Tamil morphemes
    sandhi_rules: int = 14  # 14 sandhi transformation rules
    lowercase: bool = True
    remove_punctuation: bool = False


@dataclass
class DataConfig:
    """Data processing configuration."""
    corpus_path: str = "data/corpus.jsonl"
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15
    batch_size: int = 16
    num_workers: int = 4
    
    # Scrapers
    scrapers: List[str] = field(default_factory=lambda: ["dinamalar", "bbc_tamil", "dinamani"])
    scraper_timeout: int = 30
    scraper_max_retries: int = 3
    
    # Processors
    clean_text: bool = True
    validate_corpus: bool = True
    remove_duplicates: bool = True


@dataclass
class ModelConfig:
    """Model training/inference configuration."""
    model_type: str = "xlm-roberta-base"  # Can be overridden for Gemma, etc.
    learning_rate: float = 2e-5
    num_epochs: int = 3
    warmup_steps: int = 500
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    
    # Device
    device: str = "cpu"  # "cpu", "cuda", "mps"
    fp16: bool = False
    
    # Checkpointing
    save_strategy: str = "epoch"
    eval_strategy: str = "epoch"
    save_total_limit: int = 3
    output_dir: str = "models/checkpoints"


@dataclass
class EvalConfig:
    """Evaluation metrics configuration."""
    metrics: List[str] = field(default_factory=lambda: [
        "morphological_accuracy",
        "tokenization_f1",
        "agglutination_support",
        "sandhi_preservation",
        "diglossia_handling",
        "accuracy",
        "f1",
        "perplexity"
    ])
    eval_batch_size: int = 32
    compute_metrics_frequency: int = 100


@dataclass
class Config:
    """Master configuration."""
    name: str = "adhan"
    version: str = "0.2.0"
    description: str = "Tamil Large Language Model"
    
    # Subconfigs
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    
    # Paths
    project_root: Path = field(default_factory=lambda: Path(__file__).parent.parent.parent)
    data_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent.parent / "data")
    models_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent.parent / "models")
    logs_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent.parent / "logs")
    
    # Logging
    log_level: str = "INFO"
    
    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        """Load config from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        
        # Parse nested configs
        tokenizer_data = data.pop("tokenizer", {})
        data_data = data.pop("data", {})
        model_data = data.pop("model", {})
        eval_data = data.pop("eval", {})
        
        return cls(
            tokenizer=TokenizerConfig(**tokenizer_data),
            data=DataConfig(**data_data),
            model=ModelConfig(**model_data),
            eval=EvalConfig(**eval_data),
            **data
        )
    
    def to_yaml(self, path: str) -> None:
        """Save config to YAML file."""
        data = {
            "name": self.name,
            "version": self.version,
            "tokenizer": {
                "vocab_size": self.tokenizer.vocab_size,
                "model_type": self.tokenizer.model_type,
                "max_length": self.tokenizer.max_length,
            },
            "data": {
                "corpus_path": self.data.corpus_path,
                "train_split": self.data.train_split,
                "batch_size": self.data.batch_size,
            },
            "model": {
                "model_type": self.model.model_type,
                "learning_rate": self.model.learning_rate,
                "num_epochs": self.model.num_epochs,
            },
        }
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    
    def __str__(self) -> str:
        return f"Config(name={self.name}, version={self.version}, device={self.model.device})"


def get_default_config() -> Config:
    """Get default Adhan configuration."""
    return Config()


def get_rpi5_config() -> Config:
    """Get RPi 5-optimized configuration."""
    config = Config()
    config.model.device = "cpu"
    config.model.fp16 = False
    config.data.batch_size = 8  # Smaller batch for limited VRAM
    config.data.num_workers = 2  # Fewer workers
    config.model.num_epochs = 1  # Shorter training for testing
    return config


def get_cluster_config() -> Config:
    """Get S-Node cluster configuration (4× RPi 5)."""
    config = Config()
    config.model.device = "cpu"
    config.data.batch_size = 32  # Distributed training
    config.data.num_workers = 8
    config.model.num_epochs = 3
    config.model.output_dir = "models/checkpoints-cluster"
    return config

"""
Adhan: Tamil Large Language Model

A modular, scalable LLM for Tamil with:
- Morphologically-aware tokenization (70+ morphemes, 14 sandhi rules)
- Agglutinative decomposition
- Multi-source corpus (news, colloquial, classical)
- Evaluation metrics (15 Tamil-specific metrics)
- Training pipeline (PyTorch + inference fallback)
"""

__version__ = "0.2.0"
__author__ = "Yazhi Foundation"

from .core import (
    Config,
    TokenizerConfig,
    DataConfig,
    ModelConfig,
    EvalConfig,
    get_default_config,
    get_rpi5_config,
    get_cluster_config,
)
from .tokenizer import (
    BaseTokenizer,
    TamilAgglutinativeTokenizer,
    decompose_morphemes,
)
from .data import (
    Corpus,
    CorpusSample,
    BaseScraper,
)
from .models import (
    BaseModel,
    Trainer,
    Evaluator,
    InferenceEngine,
)

__all__ = [
    "Config",
    "TokenizerConfig",
    "DataConfig",
    "ModelConfig",
    "EvalConfig",
    "get_default_config",
    "get_rpi5_config",
    "get_cluster_config",
    "BaseTokenizer",
    "TamilAgglutinativeTokenizer",
    "decompose_morphemes",
    "Corpus",
    "CorpusSample",
    "BaseScraper",
    "BaseModel",
    "Trainer",
    "Evaluator",
    "InferenceEngine",
]

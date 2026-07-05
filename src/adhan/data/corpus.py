"""
Corpus loader and manager.
Handles loading, splitting, validation of Tamil training data.
"""

import json
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from ..core import DataConfig, setup_logger

logger = setup_logger(__name__)


@dataclass
class CorpusSample:
    """Single corpus sample."""
    text: str
    source: str = "unknown"
    dialect: str = "classical"  # classical, colloquial, news
    metadata: Optional[Dict] = None
    
    def to_dict(self) -> Dict:
        return {
            "text": self.text,
            "source": self.source,
            "dialect": self.dialect,
            "metadata": self.metadata or {},
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "CorpusSample":
        return cls(
            text=data.get("text", ""),
            source=data.get("source", "unknown"),
            dialect=data.get("dialect", "classical"),
            metadata=data.get("metadata", {}),
        )


class Corpus:
    """
    Tamil corpus manager.
    Load, validate, split into train/val/test.
    """
    
    def __init__(self, config: DataConfig):
        self.config = config
        self.samples: List[CorpusSample] = []
        self.train_samples: List[CorpusSample] = []
        self.val_samples: List[CorpusSample] = []
        self.test_samples: List[CorpusSample] = []
    
    def load_jsonl(self, path: str) -> None:
        """Load corpus from JSONL file."""
        path_obj = Path(path)
        if not path_obj.exists():
            logger.warning(f"Corpus file not found: {path}")
            return
        
        with open(path_obj) as f:
            for line_no, line in enumerate(f, 1):
                try:
                    data = json.loads(line.strip())
                    sample = CorpusSample.from_dict(data)
                    if sample.text.strip():
                        self.samples.append(sample)
                except json.JSONDecodeError as e:
                    logger.warning(f"Line {line_no}: Invalid JSON: {e}")
                except Exception as e:
                    logger.warning(f"Line {line_no}: {e}")
        
        logger.info(f"Loaded {len(self.samples)} samples from {path}")
    
    def validate(self) -> Tuple[int, int]:
        """
        Validate corpus.
        Returns (valid_count, invalid_count).
        """
        valid = 0
        invalid = 0
        
        for sample in self.samples:
            if len(sample.text.strip()) > 10:  # Min length
                valid += 1
            else:
                invalid += 1
                logger.debug(f"Skipping short text: {sample.text[:30]}")
        
        logger.info(f"Corpus validation: {valid} valid, {invalid} invalid")
        return valid, invalid
    
    def remove_duplicates(self) -> int:
        """Remove duplicate texts from corpus."""
        seen = set()
        unique_samples = []
        duplicates = 0
        
        for sample in self.samples:
            if sample.text not in seen:
                seen.add(sample.text)
                unique_samples.append(sample)
            else:
                duplicates += 1
        
        self.samples = unique_samples
        logger.info(f"Removed {duplicates} duplicates; {len(self.samples)} remain")
        return duplicates
    
    def split(self) -> None:
        """Split corpus into train/val/test."""
        if not self.samples:
            logger.warning("Cannot split empty corpus")
            return
        
        n = len(self.samples)
        train_end = int(n * self.config.train_split)
        val_end = train_end + int(n * self.config.val_split)
        
        self.train_samples = self.samples[:train_end]
        self.val_samples = self.samples[train_end:val_end]
        self.test_samples = self.samples[val_end:]
        
        logger.info(f"Split: train={len(self.train_samples)}, "
                   f"val={len(self.val_samples)}, test={len(self.test_samples)}")
    
    def save_split(self, output_dir: str) -> None:
        """Save train/val/test splits to separate JSONL files."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        splits = {
            "train.jsonl": self.train_samples,
            "val.jsonl": self.val_samples,
            "test.jsonl": self.test_samples,
        }
        
        for filename, samples in splits.items():
            path = output_dir / filename
            with open(path, "w") as f:
                for sample in samples:
                    f.write(json.dumps(sample.to_dict(), ensure_ascii=False) + "\n")
            logger.info(f"Saved {len(samples)} samples to {path}")
    
    def get_texts(self, split: str = "train") -> List[str]:
        """Get list of texts for training."""
        if split == "train":
            samples = self.train_samples
        elif split == "val":
            samples = self.val_samples
        elif split == "test":
            samples = self.test_samples
        else:
            samples = self.samples
        
        return [s.text for s in samples]
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __repr__(self) -> str:
        return f"Corpus(size={len(self)}, train={len(self.train_samples)}, val={len(self.val_samples)}, test={len(self.test_samples)})"

# Adhan: Tamil Large Language Model (Modular Architecture)

**Version:** 0.2.0 | **Status:** Cycle 8 Phase 1-3 (In Progress)

## Overview

Adhan is a modular, scalable Tamil LLM with:
- **Morphologically-aware tokenization** — 70+ Tamil morphemes, 14 sandhi transformation rules
- **Agglutination-aware decomposition** — Handles Tamil's SOV word order, pro-drop phenomena
- **Multi-source corpus** — News (Dinamalar, BBC Tamil, Dinamani), colloquial, classical
- **Performance-focused** — Optimized for RPi 5, S-Node cluster, and PyTorch
- **Production-ready** — Installable package, CLI, comprehensive test suite

## Quick Start

### Installation (Development)

```bash
cd models/adhan
pip install -e .
```

### Usage

```python
from adhan import (
    TamilAgglutinativeTokenizer,
    Corpus,
    Config,
    get_rpi5_config,
)

# Load config
config = get_rpi5_config()

# Create tokenizer
tokenizer = TamilAgglutinativeTokenizer(config.tokenizer)

# Tokenize Tamil text
text = "நான் வீட்டிற்கு செல்கிறேன்"
tokens = tokenizer.tokenize(text)
encoded = tokenizer.encode(text)

# Load corpus
corpus = Corpus(config.data)
corpus.load_jsonl("data/combined_corpus.jsonl")
corpus.validate()
corpus.split()
```

## Architecture

```
src/adhan/
├── core/           # Configuration, logging, exceptions
│   ├── config.py   # Unified multi-profile configs (default, rpi5, cluster)
│   ├── exceptions.py
│   └── logger.py
│
├── tokenizer/      # Tamil-aware tokenization (70+ morphemes)
│   ├── base.py     # Abstract BaseTokenizer
│   ├── tamil_agglutinative.py  # Main implementation
│   └── utils.py    # Morpheme decomposition, sandhi rules
│
├── data/           # Corpus, scraping, preprocessing
│   ├── corpus.py   # Corpus class (load, validate, split)
│   ├── scrapers/
│   │   ├── base.py
│   │   ├── dinamalar.py
│   │   ├── bbc_tamil.py
│   │   └── dinamani.py
│   └── processors/
│       ├── cleaner.py
│       └── validator.py
│
├── models/         # Training, evaluation, inference
│   ├── base.py
│   ├── train.py
│   ├── evaluate.py
│   └── inference.py
│
└── cli/            # Command-line interface
    ├── train.py    # $ adhan train
    ├── eval.py     # $ adhan eval
    ├── scrape.py   # $ adhan scrape
    └── generate.py # $ adhan generate
```

## Configuration

Three profiles provided:

### Default (Development)

```yaml
# config/default.yaml
model:
  device: cpu
  batch_size: 16
  num_epochs: 3
```

**Usage:** `adhan train` (uses default config)

### RPi 5 (Optimized)

```yaml
# config/rpi5.yaml
model:
  device: cpu
  batch_size: 8      # Limited VRAM
  num_workers: 2
  num_epochs: 1      # Quick test
```

**Usage:** `adhan train --config config/rpi5.yaml`

### Cluster (S-Node × 4 RPi 5)

```yaml
# config/cluster.yaml
distributed:
  enabled: true
  num_nodes: 4
  
model:
  batch_size: 32    # Distributed batch
  num_workers: 8
```

**Usage:** `adhan train --config config/cluster.yaml` (on coordinating node)

## Core Components

### Tokenizer: TamilAgglutinativeTokenizer

```python
from adhan import TamilAgglutinativeTokenizer, TokenizerConfig

config = TokenizerConfig(vocab_size=32000, morphemes=70, sandhi_rules=14)
tokenizer = TamilAgglutinativeTokenizer(config)

# Tokenize + encode
tokens = tokenizer.tokenize("நான் செல்கிறேன்")
encoded = tokenizer.encode("நான் செல்கிறேன்")

# Morphological analysis  
analysis = tokenizer.get_morphological_analysis("நான் செல்கிறேன்")
# {
#   "words": [
#     {
#       "word": "நான்",
#       "morphemes": [("நான்", "root")],
#       "features": {"person": "first_person", "number": "singular"}
#     }
#   ]
# }
```

### Corpus: Load & Manage Training Data

```python
from adhan import Corpus, DataConfig

config = DataConfig()
corpus = Corpus(config)

# Load from JSONL
corpus.load_jsonl("data/combined_corpus.jsonl")

# Validate & clean
valid, invalid = corpus.validate()
corpus.remove_duplicates()

# Split into train/val/test
corpus.split()

# Save splits
corpus.save_split("data/splits")

# Get texts for training
train_texts = corpus.get_texts(split="train")
```

### Config: Multi-profile Management

```python
from adhan import get_default_config, get_rpi5_config, get_cluster_config

# Get profile
config = get_rpi5_config()

# Override specific values
config.model.learning_rate = 1e-5
config.data.batch_size = 4

# Load from YAML
from adhan import Config
config = Config.from_yaml("config/custom.yaml")

# Save config
config.to_yaml("config/export.yaml")
```

## Development Status

### ✅ Completed (Cycle 8)

- **Phase 1:** Core infrastructure (Config, Exceptions, Logging)
- **Phase 2:** Tokenizer module (BaseTokenizer, TamilAgglutinativeTokenizer, morphological utilities)
- **Phase 3:** Data layer (Corpus, BaseScraper, data loading)

### 🚧 In Progress (Cycle 8)

- **Phase 4:** Model layer (BaseModel, Trainer, Evaluator, Inference)
- **Phase 5:** CLI (train, scrape, eval, generate commands)
- **Phase 6:** Documentation + full test suite

### ⏳ Planned (Cycle 8+)

- Concrete scrapers (Dinamalar, BBC Tamil, Dinamani)
- Concrete text processors (cleaner, validator)
- Full integration tests
- Performance benchmarks
- S-Node cluster training

## Design Principles

1. **Single Responsibility** — Each module does ONE thing
2. **Dependency Injection** — Pass config objects, no globals
3. **Testability** — All classes mockable and independent
4. **Scalability** — Config-driven (same code for laptop → cluster)
5. **Backward Compatibility** — Old scripts still work; new structure coexists

## Testing

```bash
cd models/adhan

# Unit tests
pytest tests/test_tokenizer.py -v
pytest tests/test_data.py -v

# Integration test
python -c "from adhan import TamilAgglutinativeTokenizer, Corpus; print('✅ All imports OK')"

# Full test suite (coming Cycle 8 Phase 6)
pytest tests/ -v --cov=src/adhan
```

## File Structure

```
models/adhan/
├── src/adhan/                  # Main package (installable)
│   ├── __init__.py
│   ├── core/
│   ├── tokenizer/
│   ├── data/
│   ├── models/
│   └── cli/
│
├── tests/                      # Test suite
│   ├── test_tokenizer.py
│   ├── test_data.py
│   ├── test_models.py
│   └── fixtures/               # Sample data
│
├── config/                     # Deployment profiles
│   ├── default.yaml
│   ├── rpi5.yaml
│   └── cluster.yaml
│
├── setup.py                    # Installable package
├── README.md                   # This file
└── scripts/                    # Entry points (thin wrappers)
    ├── train
    ├── eval
    ├── scrape
    └── generate
```

## Migration from Old Structure

**Old structure:** `/scripts/*.py` (flat)
**New structure:** `src/adhan/{core,tokenizer,data,models,cli}` (hierarchical)

Old scripts still work but are now **thin wrappers**:

```bash
# Old way (still works)
python scripts/train_adhan_real.py --steps 100

# New way (recommended)
adhan train --config config/rpi5.yaml --steps 100
```

## Next Steps (Cycle 8 Phases 4-5)

1. **Model layer** (train.py, evaluate.py, inference.py)
2. **CLI** (Click/argparse commands)
3. **Full test suite** (pytest fixtures, mocks)
4. **GitHub integration** (push to yazhi-lem/adhan)

## References

- **Tokenizer:** `/src/adhan/tokenizer/utils.py` (70+ Tamil morphemes, 14 sandhi rules)
- **Config:** `/config/default.yaml`, `rpi5.yaml`, `cluster.yaml`
- **Refactor Plan:** `~/.hermes/plans/adhan-refactor-cycle8.md`

## License

MIT — See LICENSE file

---

**Last Updated:** 2026-07-05  
**Cycle:** Rotation 26, Cycle 8 Phase 1-3  
**Status:** Modular architecture ready for training/evaluation layers

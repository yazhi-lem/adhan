# Adhan

Tamil-first LLM data + training pipeline.

## 🆕 Native Tamil SLM (swaram tokens + JAX)

We are building a **from-scratch, pure-Tamil small language model** — akshara
(உயிர்–மெய் / *swaram*) as the atomic token, agglutination-aware modeling, trained in
**JAX/Flax** and tracked with **MLflow**, targeting a light, edge-deployable launch.

- **Roadmap:** [`ROADMAP_JAX_SLM.md`](ROADMAP_JAX_SLM.md)
- **Architecture:** [`docs/ARCHITECTURE_SWARAM_SLM.md`](docs/ARCHITECTURE_SWARAM_SLM.md)
- **Code:** [`src/adhan_slm/`](src/adhan_slm/) (working swaram tokenizer + Flax SLM + JAX/MLflow trainer)

```bash
PYTHONPATH=src python -m adhan_slm.tokenizer.swaram_tokenizer "படித்துக்கொண்டிருந்தேன்"
```

The existing PyTorch pipeline below is reused for corpus building and as baselines.

## Recent changes

- Added shared constants in `src/core/`
- Added corpus merger: `src/data_scraper/merge_corpora.py`
- Added Gemma training notebook: `src/notebooks/03_gemma_training.ipynb`

## Installation

### Option 1: Development Install (Recommended)

```bash
# Clone the repository
git clone https://github.com/yazhi-lem/adhan.git
cd adhan

# Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install the package in development mode with all dependencies
pip install -e ".[dev,jax,tamil-nlp]"
```

### Option 2: JAX Stack Only (for training)

```bash
pip install -e ".[jax]"
```

### Option 3: PyTorch Stack Only (for baselines)

```bash
pip install -e ".[pytorch]"
```

## Quick Start

**Test the swaram tokenizer** (no JAX/PyTorch needed):
```bash
python -m adhan_slm.tokenizer.swaram_tokenizer "படித்துக்கொண்டிருந்தேன்"
```

**Run unit tests** (ensure everything works):
```bash
pytest tests/ -v
```

**Try the full pipeline** (after installing JAX):
```bash
# 1. Prepare corpus and freeze tokenizer
python scripts/prepare_slm_corpus.py \
    --corpus data/raw/tamil/ --out data/final/tamil_slm \
    --vocab-size 12000 --seq-len 1024

# 2. Train a model (smoke test)
python -m adhan_slm.training.train_jax \
    --config src/adhan_slm/configs/adhan_slm_tiny.yaml --smoke

# 3. Generate text from a checkpoint
python scripts/generate_slm.py \
    --tokenizer-dir data/final/tamil_slm \
    --checkpoint checkpoints/adhan-tiny \
    --prompt "சொல், உனக்கு பிடித்த உணவு என்ன?"

# 4. Run full evaluation suite
python -m adhan_slm.eval.run_eval \
    --tokenizer-dir data/final/tamil_slm \
    --config src/adhan_slm/configs/adhan_slm_tiny.yaml \
    --checkpoint checkpoints/adhan-tiny
```

## Documentation

- **[Roadmap](ROADMAP_JAX_SLM.md)** — Phased development plan (Phase 0 done, Phase A in progress)
- **[Architecture](docs/ARCHITECTURE_SWARAM_SLM.md)** — Swaram tokenizer + JAX/Flax model design
- **[Completion Tracker](docs/COMPLETION_TRACKER.md)** — Real-time progress on all phases
- **[Phase A Tracker](docs/PHASE_A_TRACKER.md)** — Current work (CI/CD, logging, packaging)

## Run scripts

Use dedicated scripts:

- `scripts/run_scraper.py` for corpus build + HF export
- `scripts/run_training.py` for training
- `scripts/run_model.py` for full orchestration

```bash
# Build corpus + export HF splits
python scripts/run_scraper.py --strategy modern --max-records 80000

# Train model
python scripts/run_training.py --num-epochs 3 --batch-size 4

# Full run (build + train, optional merge)
python scripts/run_model.py --strategy modern --num-epochs 3 --batch-size 4
```

For full command sequence and examples, see `DEV.md`.

## Core scripts

- `src/data_scraper/processing/build_unified_corpus.py`
- `src/data_scraper/export/export_unified_hf.py`
- `src/data_scraper/merge_corpora.py`
- `src/models/sangam_gpt/train_enhanced.py`
- `scripts/run_scraper.py`
- `scripts/run_training.py`
- `scripts/run_model.py`

## License

MIT

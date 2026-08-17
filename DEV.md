# DEV Commands

Minimal command sequence for local development.

## 1) Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,jax,tamil-nlp,data-collection]"
```

## 2) Build corpus (Phase 2 data collection)

```bash
python scripts/run_scraper.py \
  --strategy modern \
  --data-dir data/raw \
  --existing-corpus data/intermediate/rebalancing/v3_modern_enhanced.jsonl \
  --corpus-output data/intermediate/rebalancing/unified_modern.jsonl \
  --hf-output data/final/tamil_texts/hf \
  --max-records 80000
```

## 3) Freeze tokenizer + pack shards

```bash
python scripts/prepare_slm_corpus.py \
  --corpus data/raw/tamil/ --out data/final/tamil_slm \
  --vocab-size 12000 --seq-len 1024 --val-frac 0.02
```

## 4) Train (JAX)

```bash
python -m adhan_slm.training.train_jax \
  --config src/adhan_slm/configs/adhan_slm_tiny.yaml --smoke
```

## 5) Generate from a checkpoint

```bash
python scripts/generate_slm.py \
  --tokenizer-dir data/final/tamil_slm \
  --config src/adhan_slm/configs/adhan_slm_tiny.yaml \
  --checkpoint checkpoints/nano \
  --prompt "சொல், உனக்கு பிடித்த உணவு என்ன?" --temperature 0.8
```

## 6) Optional corpus merge (ADHAN + VAZHI)

```bash
python src/data_scraper/merge_corpora.py \
  --adhan_dir data/final/tamil_texts/hf \
  --vazhi_repo ../vazhi \
  --output data/unified/tamil_6k.jsonl \
  --target_count 6000 \
  --split
```

## 7) Social media collection (Phase 2)

```bash
# Reddit only
python scripts/run_scraper.py --social reddit --social-max-posts 500

# Twitter/X only
python scripts/run_scraper.py --social twitter --social-max-requests 100

# Both + corpus rebuild
python scripts/run_scraper.py --social all
```

## 8) Native SLM: CPU training (no GPU)

Full guide: [`docs/CPU_TRAINING.md`](docs/CPU_TRAINING.md).

```bash
# Install the CPU JAX stack (GPU is opt-in via ".[jax-cuda]")
pip install -e ".[jax,dev]"

# 1. Freeze the swaram tokenizer + pack shards at the CPU sequence length
python scripts/prepare_slm_corpus.py \
  --corpus data/raw/tamil/ \
  --out data/final/tamil_slm_cpu \
  --vocab-size 4000 --seq-len 256

# 2. Sanity-gate the wiring: one batch, repeated, loss must collapse
python -m adhan_slm.training.train_jax \
  --config src/adhan_slm/configs/adhan_slm_nano_cpu.yaml \
  --device cpu --overfit-batch

# 3. Train (resumable; re-running continues the global step budget)
python -m adhan_slm.training.train_jax \
  --config src/adhan_slm/configs/adhan_slm_nano_cpu.yaml --device cpu

# 4. Inspect runs
mlflow ui   # http://localhost:5000
```

Useful overrides (no config edit needed):

```bash
--max-steps 500 --batch-size 4 --grad-accum-steps 16 --learning-rate 1e-3
```

Tests for the CPU path (~1 min, no GPU):

```bash
pytest tests/integration/train_cpu_*_tests.py -v   # ~3 min (XLA compiles per config)
python -m adhan_slm.training.device_tests          # dtype/backend resolution, <1s
```

## Notes

- Run from repository root.
- Use `python scripts/run_scraper.py --help`, `python scripts/prepare_slm_corpus.py --help`,
  and `python scripts/generate_slm.py --help`.
- The legacy PyTorch fine-tuning pipeline (Gemma LoRA, XLM-RoBERTa MLM,
  sangam_gpt, sentiment fine-tuning) has been removed — the project now
  stands on the from-scratch JAX SLM plus this Phase 2 data-collection
  pipeline only. See `ROADMAP_JAX_SLM.md` §0 for the rationale.
- `scripts/export_onnx.py` / `scripts/quantize_model.py` still exist but are
  written against HF/transformers model dirs from the removed pipeline; they
  need reworking for the JAX SLM before they're usable again.

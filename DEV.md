# DEV Commands

Minimal command sequence for local development.

## 1) Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2) Build corpus (dependency order)

```bash
python scripts/run_scraper.py \
  --strategy modern \
  --data-dir data/raw \
  --existing-corpus data/intermediate/rebalancing/v3_modern_enhanced.jsonl \
  --corpus-output data/intermediate/rebalancing/unified_modern.jsonl \
  --hf-output data/final/tamil_texts/hf \
  --max-records 80000
```

## 3) Train

```bash
python scripts/run_training.py \
  --data-dir data/final/tamil_texts/hf \
  --output-dir models/adhan \
  --num-epochs 3 \
  --batch-size 4 \
  --learning-rate 5e-5
```

## 4) Optional merge (ADHAN + VAZHI)

```bash
python src/data_scraper/merge_corpora.py \
  --adhan_dir data/final/tamil_texts/hf \
  --vazhi_repo ../vazhi \
  --output data/unified/tamil_6k.jsonl \
  --target_count 6000 \
  --split
```

## 5) One command (build + train)

```bash
python scripts/run_model.py \
  --strategy modern \
  --max-records 80000 \
  --num-epochs 3 \
  --batch-size 4
```

## Notes

- Run from repository root.
- Use `python scripts/run_scraper.py --help`, `python scripts/run_training.py --help`, and `python scripts/run_model.py --help`.
- Current runner executes:
  - `build_unified_corpus.py`
  - `export_unified_hf.py`
  - `train_enhanced.py`
  - `src/data_scraper/merge_corpora.py` (optional)

## 6) Social media collection (Phase 2)

```bash
# Reddit only
python scripts/run_scraper.py --social reddit --social-max-posts 500

# Twitter/X only
python scripts/run_scraper.py --social twitter --social-max-requests 100

# Both + corpus rebuild
python scripts/run_scraper.py --social all
```

## 7) ONNX export + quantization (Phase 3)

```bash
# Export trained model to ONNX
python scripts/export_onnx.py \
  --model-dir models/adhan \
  --output-dir models/adhan_onnx

# INT8 dynamic quantization (no calibration data required)
python scripts/quantize_model.py \
  --model-dir models/adhan_onnx \
  --mode int8-dynamic \
  --benchmark

# INT4 weight-only quantization (requires optimum[onnxruntime])
python scripts/quantize_model.py \
  --model-dir models/adhan_onnx \
  --mode int4
```

## 8) Sentiment analysis fine-tuning (Phase 4)

```bash
python src/models/sentiment/train_sentiment.py \
  --train-file data/sentiment/train.jsonl \
  --val-file   data/sentiment/val.jsonl \
  --model-name xlm-roberta-base \
  --output-dir models/adhan_sentiment \
  --num-labels 2 \
  --num-epochs 5
```

## 9) Native SLM: CPU training (no GPU)

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

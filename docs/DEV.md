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

### 2b) Clean the corpus before packing (optional, recommended)

Near-duplicate removal and benchmark-leak removal, both with machine-readable reports.
Empty corpora are fine here: pass `--max-docs` to try it on a slice first.

```bash
# MinHash near-duplicates (O(n) via LSH banding; 128 hashes / 5-char shingles / 0.85)
python -m adhan_slm.data.minhash_dedup data/intermediate/unified.jsonl \
  --output data/intermediate/unified.dedup.jsonl \
  --clusters data/intermediate/dupes.json

# Benchmark/eval leak removal (exact + word 8-gram + char 5-gram)
python -m adhan_slm.data.contamination build-index 'data/benchmarks/*.jsonl' \
  --output data/intermediate/benchmarks.idx.json
python -m adhan_slm.data.contamination scan \
  --index data/intermediate/benchmarks.idx.json \
  --corpus data/intermediate/unified.dedup.jsonl \
  --output data/intermediate/unified.clean.jsonl \
  --report data/intermediate/contamination.json
```

After `pip install -e .` the same two tools are the console scripts `adhan-dedup` and
`adhan-decontaminate`. Record the removed-doc counts in the shard `datasheet.json`.

## 3) Freeze tokenizer + pack shards

```bash
python scripts/prepare_slm_corpus.py \
  --corpus data/raw/tamil/ --out data/final/tamil_slm \
  --vocab-size 12000 --seq-len 1024 --val-frac 0.02
```

### Where the script facts live

Every Unicode table used by the tokenizer family is in
[`src/adhan_slm/common/scripts.py`](../src/adhan_slm/common/scripts.py) — one validated
registry for the ten Brahmic scripts, the Tamil letter inventory (12 uyir, 18 mei, 5
Grantha meis, 11 vowel signs, 10 digits, āytam) and the 48-column Dravidian phonetic
matrix. It validates itself on import and the tables are read-only.

```python
from adhan_slm.common.scripts import TAMIL, akshara_inventory, segment, cognates

akshara_inventory(TAMIL)        # 322 tokens; no unassigned codepoints
segment("கற்போம்", (TAMIL,))   # ['க', 'ற்', 'போ', 'ம்']
cognates("ழ")                   # {'tamil': 'ழ', 'telugu': 'ఴ', 'kannada': 'ೞ', 'malayalam': 'ഴ'}
```

Swaram, Aksharam and the multi-script path all call the same `segment`; do not add a
second segmenter. Retraining note: vocabularies built before 9 Nov 2026 carry reserved
codepoints (148 junk tokens in `data/final/tamil_slm/vocab.json`, from the old
`range(0x0BBE, 0x0BCD)` vowel-sign loop). Regenerate the vocab rather than reusing it,
and do not re-tokenize already-packed shards silently.

```bash
python -m pytest tests/test_common_scripts.py tests/test_tokenizer_hardening.py \
  src/adhan_slm/tokenizer -q     # 239 passed / 1 skipped, ~1s, no JAX
```

## 4) Train (JAX)

```bash
python -m adhan_slm.training.train_jax \
  --config src/adhan_slm/configs/adhan_slm_tiny.yaml --smoke
```

## 5) Chat with a checkpoint

```bash
python -m adhan_slm.chat_cli \
  --checkpoint checkpoints/adhan-nano-sft \
  --tokenizer data/final/tamil_slm \
  --prompt "வணக்கம்! நீ யார்?"
```

Installed as the console script `adhan-chat`. Add `--temperature 0.8 --top-k 40` for
sampling; the default is greedy. Distress phrases in the user turn trigger a Tele-MANAS
14416 / 112 footer — a wrapper safeguard, not a model capability.

## 5b) Generate from a checkpoint

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

Full guide: [`docs/training/CPU_TRAINING.md`](training/CPU_TRAINING.md).

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

## Build / release gate

```bash
./scripts/build_package.sh          # black, isort, ruff, pytest, entry points, sdist+wheel
./scripts/build_package.sh --fast   # same minus pytest
```

Needs `pip install build hatchling` for the packaging step. Override the interpreter with
`PYTHON=/path/to/python ./scripts/build_package.sh`.

## Notes

- Run from repository root.
- Use `python scripts/run_scraper.py --help`, `python scripts/prepare_slm_corpus.py --help`,
  and `python scripts/generate_slm.py --help`.
- The legacy PyTorch fine-tuning pipeline (Gemma LoRA, XLM-RoBERTa MLM,
  sangam_gpt, sentiment fine-tuning) has been removed — the project now
  stands on the from-scratch JAX SLM plus this Phase 2 data-collection
  pipeline only. See `docs/ROADMAP_JAX_SLM.md` §0 for the rationale.
- `scripts/export_onnx.py` / `scripts/quantize_model.py` still exist but are
  written against HF/transformers model dirs from the removed pipeline; they
  need reworking for the JAX SLM before they're usable again.

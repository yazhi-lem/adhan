# adhan_slm — native Tamil SLM (swaram tokens + JAX)

From-scratch, akshara-native, agglutination-aware small language model for Tamil.
Not a fine-tune of an English/multilingual base. See the top-level
[`ROADMAP_JAX_SLM.md`](../../ROADMAP_JAX_SLM.md) and
[`docs/ARCHITECTURE_SWARAM_SLM.md`](../../docs/ARCHITECTURE_SWARAM_SLM.md).

Adhan is the **foundational model** of the Yazh ecosystem — see
[`docs/YAZH_FOUNDATION.md`](../../docs/YAZH_FOUNDATION.md).

## Layout
| Path | What |
|---|---|
| `tokenizer/swaram_tokenizer.py` | **Swaram** — Dravidian/Tamil akshara (உயிர்–மெய்) segmentation + morpheme-merge BPE. Lossless, tested. |
| `tokenizer/aksharam_tokenizer.py` | **Aksharam** — Indic/Hindi Devanagari (matra + conjunct) tokenizer, sibling of Swaram. Lossless, tested. |
| `tokenizer/jax_encode.py` | JAX-accelerated batch encoding shared by both tokenizers. |
| `model/transformer.py` | Flax decoder-only SLM (RoPE + RMSNorm + SwiGLU, tied embeddings) + `generate()` (temp/top-k/top-p). `nano/tiny/mini`. |
| `data/` | Corpus → packed shards → batches: `corpus.py` (txt/jsonl/dir reader), `packing.py` (tokenize + pack + `.bin` shards), `loader.py` (deterministic batching). Pure-python core, tested. |
| `training/train_jax.py` | JAX/Flax pretraining loop: real-data iterator, in-loop val perplexity, Orbax checkpoint/resume, MLflow tracking. |
| `training/mlflow_utils.py` | MLflow run contract (params + git SHA + data version). |
| `inference.py` | Load frozen tokenizer + Orbax checkpoint → `generate_text()`. |
| `configs/adhan_slm_tiny.yaml` | Model + training + data + checkpoint config. |
| `eval/` | Tamil-first evaluation: `morphology.py` (stemmer/sandhi), `kid_level_prompts.py`, `ngram_baseline.py`, `perplexity.py` (model ppl), and `run_eval.py` (one-shot harness → JSON). See [`docs/EVAL_TAMIL.md`](../../docs/EVAL_TAMIL.md). |
| `external/open_tamil_bridge.py` | Bridge onto **open-tamil** (MIT), the base Tamil-NLP layer for eval/tooling — segmentation oracle, stemmer, sandhi checker, encoding/transliteration, lexicons. Never imported by the tokenizer hot path. |
| `../../scripts/prepare_slm_corpus.py` | Freeze tokenizer (`vocab.json`+`merges.txt`) + pack shards + `datasheet.json`. |
| `../../scripts/generate_slm.py` | Sample text from a trained checkpoint. |

## Quick start
```bash
# tokenizers work with zero extra deps (pure python)
PYTHONPATH=src python -m adhan_slm.tokenizer.swaram_tokenizer "படித்துக்கொண்டிருந்தேன்"   # Tamil
PYTHONPATH=src python -m adhan_slm.tokenizer.aksharam_tokenizer "पढ़ रहा था"               # Hindi
PYTHONPATH=src python src/adhan_slm/tokenizer/test_swaram_tokenizer.py     # 5 tests
PYTHONPATH=src python src/adhan_slm/tokenizer/test_aksharam_tokenizer.py   # 5 tests

# model size table (no JAX needed for configs)
PYTHONPATH=src python -m adhan_slm.model.transformer

# data pipeline unit tests (pure-python, no JAX/numpy needed)
PYTHONPATH=src python -m adhan_slm.data.test_data_pipeline

# open-tamil-backed eval (needs: pip install open-tamil, or requirements-jax.txt)
PYTHONPATH=src python src/adhan_slm/tokenizer/test_open_tamil_crosscheck.py  # segmenter cross-check
PYTHONPATH=src python src/adhan_slm/eval/test_open_tamil_eval.py            # morphology/prompts/baseline
PYTHONPATH=src python -m adhan_slm.eval.kid_level_prompts                   # preview 10 kid-level prompts

# full pipeline needs the JAX stack
python3 -m venv .venv-jax && source .venv-jax/bin/activate
pip install -r requirements-jax.txt
# 1. freeze tokenizer + pack corpus into train/val shards
python scripts/prepare_slm_corpus.py --corpus data/raw/tamil/ --out data/final/tamil_slm
# 2. smoke-train, then a real run on the shards (checkpoints + val ppl to MLflow)
PYTHONPATH=src python -m adhan_slm.training.train_jax --config src/adhan_slm/configs/adhan_slm_tiny.yaml --smoke
PYTHONPATH=src python -m adhan_slm.training.train_jax --config src/adhan_slm/configs/adhan_slm_tiny.yaml
# 3. generate + evaluate
python scripts/generate_slm.py --tokenizer-dir data/final/tamil_slm --config src/adhan_slm/configs/adhan_slm_tiny.yaml --checkpoint checkpoints/adhan-tiny --prompt "சொல், உனக்கு பிடித்த உணவு என்ன?"
PYTHONPATH=src python -m adhan_slm.eval.run_eval --tokenizer-dir data/final/tamil_slm --eval-text data/final/tamil_slm --config src/adhan_slm/configs/adhan_slm_tiny.yaml --checkpoint checkpoints/adhan-tiny
mlflow ui   # http://localhost:5000
```

## Status
- ✅ Swaram tokenizer: Layer A (akshara segmentation, lossless) + Layer B (merge BPE), tested.
- ✅ Flax SLM (three sizes) + `generate()`; jit-ed, MLflow-tracked training loop.
- ✅ **Data pipeline**: corpus → packed fixed-length shards → deterministic batches (tested).
- ✅ **Training**: real-data iterator, in-loop validation perplexity, Orbax checkpoint/resume
  with best-by-val — verified end-to-end on CPU (loss ↓, val-ppl ↓, resume, generate).
- ✅ **Corpus prep + eval harness**: `prepare_slm_corpus.py` (freeze + pack + datasheet),
  `eval/run_eval.py` (model ppl vs classical floor + morphology + kid prompts → JSON).
- ✅ open-tamil base Tamil-NLP layer: segmenter cross-check, stemmer/sandhi probes,
  kid-level prompt lexicon, classical n-gram baseline.
- 📋 Next: curate the full corpus, freeze `adhan-tok-v1`, run the full `adhan-nano`
  pretrain on GPU, and add a real distilgpt2-Tamil comparison.

Everything here is additive to the existing PyTorch corpus/fine-tune pipeline, which
is reused for data building and as baselines to beat.

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
| `model/transformer.py` | Flax decoder-only SLM (RoPE + RMSNorm + SwiGLU, tied embeddings). `nano/tiny/mini`. |
| `training/train_jax.py` | JAX/Flax pretraining loop with MLflow tracking. |
| `training/mlflow_utils.py` | MLflow run contract (params + git SHA + data version). |
| `configs/adhan_slm_tiny.yaml` | Model + training + data config. |
| `eval/` | Tamil-first evaluation: `morphology.py` (stemmer/sandhi probes), `kid_level_prompts.py`, `ngram_baseline.py` — Phase 4. |
| `external/open_tamil_bridge.py` | Bridge onto **open-tamil** (MIT), the base Tamil-NLP layer for eval/tooling — segmentation oracle, stemmer, sandhi checker, encoding/transliteration, lexicons. Never imported by the tokenizer hot path. |

## Quick start
```bash
# tokenizers work with zero extra deps (pure python)
PYTHONPATH=src python -m adhan_slm.tokenizer.swaram_tokenizer "படித்துக்கொண்டிருந்தேன்"   # Tamil
PYTHONPATH=src python -m adhan_slm.tokenizer.aksharam_tokenizer "पढ़ रहा था"               # Hindi
PYTHONPATH=src python src/adhan_slm/tokenizer/test_swaram_tokenizer.py     # 5 tests
PYTHONPATH=src python src/adhan_slm/tokenizer/test_aksharam_tokenizer.py   # 5 tests

# model size table (no JAX needed for configs)
PYTHONPATH=src python -m adhan_slm.model.transformer

# open-tamil-backed eval (needs: pip install open-tamil, or requirements-jax.txt)
PYTHONPATH=src python src/adhan_slm/tokenizer/test_open_tamil_crosscheck.py  # segmenter cross-check
PYTHONPATH=src python src/adhan_slm/eval/test_open_tamil_eval.py            # morphology/prompts/baseline
PYTHONPATH=src python -m adhan_slm.eval.kid_level_prompts                   # preview 10 kid-level prompts

# training needs the JAX stack
python3 -m venv .venv-jax && source .venv-jax/bin/activate
pip install -r requirements-jax.txt
PYTHONPATH=src python -m adhan_slm.training.train_jax \
    --config src/adhan_slm/configs/adhan_slm_tiny.yaml --smoke
mlflow ui   # http://localhost:5000
```

## Status (Phase 0 — foundation)
- ✅ Swaram tokenizer: Layer A (akshara segmentation, lossless) + Layer B (merge BPE), tested.
- ✅ Flax SLM skeleton with three sizes and a jit-ed MLflow-tracked training loop.
- ✅ open-tamil adopted as base Tamil-NLP layer: segmenter cross-check, stemmer/sandhi
  morphology probes, kid-level prompt lexicon, classical n-gram baseline.
- 📋 Next: freeze tokenizer v1 on full corpus, build packed shards, pretrain `adhan-nano`.

Everything here is additive to the existing PyTorch corpus/fine-tune pipeline, which
is reused for data building and as baselines to beat.

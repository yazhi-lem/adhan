# Adhan · Native Tamil SLM Roadmap (JAX + Swaram Tokens)

> ஆதன் — தமிழ் முதல், தமிழ் என்றும்
>
> A **native, pure-Tamil small language model (SLM)** trained from scratch — not a
> fine-tune of an English/multilingual base. It treats Tamil as Tamil: the
> **swaram / akshara** (உயிர்–மெய்) is the atomic token, and the model architecture
> and data pipeline are built around Tamil's **agglutinative** morphology. Stack is
> **JAX/Flax** for training and **MLflow** for experiment tracking. Target: a small,
> light, edge-deployable model launched **soon**.

---

## 0. Why this pivot

The current repo fine-tunes borrowed bases (Gemma LoRA, XLM-RoBERTa, distilgpt2)
with byte-level/BPE tokenizers. Those tokenizers were fit on English-dominant text
and **fragment Tamil aksharas into 3–6 sub-word pieces**, wasting sequence budget and
blurring morpheme boundaries. Tamil is agglutinative: one word (e.g.
*படித்துக்கொண்டிருந்தேன்* — "I was reading") packs root + aspect + tense + person.
A tokenizer that respects akshara + morpheme boundaries gives a **shorter, cleaner
sequence** and lets a *small* model punch above its parameter count.

**Thesis:** akshara-native tokenization + morphology-aware modeling → a genuinely
small (≤ 60M param) Tamil model that is fluent, cheap to train, and runs on a phone.

**Broader vision:** **Swaram** is a **Dravidian-language tokenizer prototype** (Tamil first,
then Kannada, Telugu, Malayalam). Parallel to it, **Aksharam** is the **Hindi prototype**
(akin to Swaram, but for Indic scripts with matras). Both feed native SLMs (Adhan for
Tamil/Dravidian, sibling models for Hindi/Indic). All ship via **yazhi-api** (the
internal deployment platform).

---

## 1. The three core ideas

### 1a. Swaram as token (uyir–mey native tokenization)
Tamil script is an abugida. The natural atomic unit is the **grapheme cluster
(akshara)**:
- **உயிர் (uyir)** — 12 vowels ("swaram", the life/sound letters)
- **மெய் (mey)** — 18 pure consonants (base + pulli ்)
- **உயிர்மெய் (uyirmey)** — 216 consonant+vowel combinations
- **ஆய்தம் (aytham)** ஃ, plus Grantha letters for loanwords (ஜ ஷ ஸ ஹ க்ஷ ஶ)

The base vocabulary is a closed, well-defined set (~250 aksharas + digits +
punctuation + control tokens). On top of the akshara layer we learn a **thin
morpheme-merge layer** (bounded BPE over aksharas) so frequent suffix chains
(-கிற-, -ந்த-, -ஏன், -ஆக, -இல்) become single tokens. Vocabulary target:
**8k–16k**, vs 250k for a multilingual base.

### 1b. Agglutination-aware modeling
- **Morphological segmentation** as a first-class preprocessing step (root | suffixes).
- **Boundary-aware positional signal** so the model sees morpheme edges.
- Optional **factored embeddings** (root embedding ⊕ suffix-role embedding) to share
  parameters across the combinatorial suffix space.
- Evaluation includes a **sandhi (புணர்ச்சி) / morphology probe**, not just perplexity.

### 1c. Small, light, launched soon
Three sizes, ship the smallest first:
| Tier | Params | d_model | layers | heads | ctx | Use |
|------|--------|---------|--------|-------|-----|-----|
| `adhan-nano` | ~8M | 256 | 6 | 4 | 512 | phone / RPi, first launch |
| `adhan-tiny` | ~33M | 512 | 8 | 8 | 1024 | default quality tier |
| `adhan-mini` | ~97M | 768 | 12 | 12 | 2048 | server / quality ceiling |

(Counts from `AdhanConfig.approx_params()` at the default swaram vocab; they scale
with vocab size since embeddings are tied.)

---

## 2. Technology stack

| Concern | Choice | Notes |
|---|---|---|
| Model / autodiff | **JAX + Flax (`nnx` or `linen`)** | `jax.jit` + `pjit`/`shard_map` for TPU/multi-GPU |
| Optimizer | **Optax** (AdamW + cosine, warmup) | gradient clipping, weight decay |
| Data | **Grain** or `tf.data` → packed streaming | akshara-tokenized shards (Arrow/TFRecord) |
| Checkpoints | **Orbax** | async, sharded, resumable |
| Experiment tracking | **MLflow** | params, metrics, artifacts, model registry |
| Tokenizer | **custom swaram tokenizer** (this repo) | pure-python core, no external base |
| Tamil-NLP foundation | **open-tamil** (MIT, Ezhil Language Foundation) | segmentation oracle, stemmer, sandhi checker, encoding/transliteration, lexicons — eval/tooling only, never the hot path |
| Serving | ONNX / GGUF / TFLite export + INT8/INT4 | reuse existing `scripts/quantize_model.py` path |

JAX is additive — it lives in `src/adhan_slm/` and `requirements-jax.txt`; the
existing PyTorch pipeline stays intact for corpus building and baselines.

---

## 3. Phased roadmap

Legend: 🎯 milestone · 📦 deliverable · ✅ done · 🚧 in progress · 📋 planned

### Phase 0 — Foundation & scaffolding  ✅ (this PR)
- ✅ 📦 `src/adhan_slm/` package: tokenizer, model, training, configs, eval
- ✅ 📦 Working **Swaram tokenizer** — Dravidian/Tamil (akshara segmentation + merges), tested
- ✅ 📦 Working **Aksharam tokenizer** — Indic/Hindi Devanagari (conjuncts + matras), tested
- ✅ 📦 **JAX-accelerated batch encoding** shared by both tokenizers (`jax_encode.py`)
- ✅ 📦 Flax transformer SLM skeleton (`adhan-nano/tiny/mini` configs)
- ✅ 📦 JAX training loop wired to **MLflow** (params/metrics/artifacts)
- ✅ 📦 **Yazh foundation** wired as base requirement (`docs/YAZH_FOUNDATION.md`); Adhan = foundational model
- ✅ 📦 `requirements-jax.txt`, architecture doc, this roadmap

### Phase 1 — Tokenizer to production  🚧  (Week 1–2)
- ✅ 📦 **open-tamil adopted as base Tamil-NLP layer** (`src/adhan_slm/external/open_tamil_bridge.py`):
  reference akshara segmenter (differential-tested against Layer A), `TamilStemmer` +
  `tamilsandhi` as the **morphological analyzer** Phase 1 called for, 25-encoding
  detector, Azhagi transliteration, `tamil.numeral`, and solthiruthi lexicons —
  see `docs/ARCHITECTURE_SWARAM_SLM.md` §10
- ✅ 📦 **Corpus-prep + freeze pipeline** (`scripts/prepare_slm_corpus.py`): trains the
  swaram/aksharam merge layer on a corpus, freezes `vocab.json` + `merges.txt`, measures
  held-out fertility against the < 1.15 target, and writes a `datasheet.json` (sources,
  token counts, code SHA) — the data card Phase 2 also asks for
- 🎯 **Fertility target: < 1.15 tokens per akshara**, round-trip lossless on held-out text
  (fertility now measured + reported by the prep script)
- 📋 Grantha + aytham + numeral + code-switch (Latin) handling; NFC normalization pass
- 📋 Run the freeze on the full curated corpus; log `adhan-tok-v1` to the MLflow registry

### Phase 2 — Corpus at pretraining scale  🚧  (Week 2–4)
- Reuse existing scrapers (`src/data_scraper/*`) + Reddit/Twitter collectors
- ✅ 📦 **Data pipeline** (`src/adhan_slm/data/`): `corpus.py` reads txt/jsonl/dir into a
  clean NFC document stream; `packing.py` tokenizes + packs into full fixed-length
  sequences (no padding) and writes flat `.bin` shards + manifests; `loader.py` yields
  deterministic, seed-shuffled batches. Pure-python core (numpy optional), unit-tested
  in `data/test_data_pipeline.py`. This is the concrete, Grain-free "tokenize → packed
  fixed-length sequences → sharded" path — it replaces the old `NotImplementedError`.
- 📋 Dedup (MinHash/LSH), quality + language-ID filter, PII scrub
- 📋 Target **300M–1B clean Tamil tokens** (nano/tiny need far less than a big LLM)
- 🎯 Frozen `adhan-corpus-v1` with a datasheet (sources, sizes, licenses) — datasheet
  emitter is wired (`prepare_slm_corpus.py`); remaining work is curating the sources
- 📦 Data card in `docs/`, splits registered in MLflow

### Phase 3 — Pretrain `adhan-nano`  🚧  (Week 4–6)
- ✅ 📦 **Single-GPU throughput pass**: fused `jax.nn.dot_product_attention`
  (auto-dispatches to a cuDNN flash-attention kernel on GPU, `is_causal=True` —
  no materialized O(T²) mask/score array), `donate_argnums` on the jit-ed train
  step (in-place state-buffer reuse), batched host sync for metric logging
  (was blocking every step on `float(loss)`, serializing XLA's async
  dispatch — now one sync per `log_every` window); XLA fallback path kept for
  older jax installs. `tokens_per_sec` now logged alongside loss/ppl/LR.
- ✅ 📦 **Real data wired into the loop**: `train_jax.data_iterator()` streams the packed
  shards from Phase 2 (was `NotImplementedError`); a non-smoke run trains on `train.bin`
  and validates on `val.bin`, deriving `seq_len` from the shard manifest.
- ✅ 📦 **In-loop validation + Orbax checkpointing**: periodic val-perplexity passes
  (`eval_every`), async resumable Orbax checkpoints with best-by-val retention
  (`checkpoint_every`, `keep_checkpoints`), and automatic resume from the latest step.
  Each checkpoint stores both the full `TrainState` (exact resume, incl. optimizer
  momentum) and a params-only item (clean load for inference/eval). Verified
  end-to-end on CPU: train loss ↓, val-ppl ↓, resume, and generate-from-checkpoint.
- 📋 Mixed precision (bf16, already wired) tuned on real data; Optax schedule sweep
- 📋 Overfit-a-batch sanity → 100M-token dry run → full nano run on GPU
- 🎯 **`adhan-nano` val perplexity beats a distilgpt2 baseline on Tamil** per-akshara ppl
- 📦 `adhan-nano-base` checkpoint + eval report

### Phase 4 — Evaluation & Tamil-specific probes  🚧  (Week 5–7)
- ✅ 📦 **Morphology probe** (`src/adhan_slm/eval/morphology.py`): stemmer-boundary
  agreement between Layer B merges and open-tamil's `TamilStemmer` suffix splits;
  sandhi correctness rate over generations via `tamilsandhi`
- ✅ 📦 **Kid-level prompt set seed** (`src/adhan_slm/eval/kid_level_prompts.py`):
  50 deterministic prompts from open-tamil's `solthiruthi` categorized lexicons
  (animals, objects, adjectives, adverbs, relations, pronouns, verbs) — matches
  the 5–7-year-old register better than a literary/classical corpus would
- ✅ 📦 **Classical baseline** (`src/adhan_slm/eval/ngram_baseline.py`): add-one
  smoothed unigram-over-aksharas perplexity via open-tamil's `ngram.LetterModels`
  — a near-zero-cost floor `adhan-nano` must clear before the distilgpt2 comparison
- ✅ 📦 **Model perplexity + eval harness** (`src/adhan_slm/eval/perplexity.py`,
  `eval/run_eval.py`): computes trained-model per-token val perplexity on a packed
  shard (directly comparable to the n-gram floor) and a single `run_eval` entrypoint
  ties every probe — fertility, classical baseline, morphology, model ppl, kid-level
  prompts (with generations) — into one JSON report. Each probe degrades gracefully
  when open-tamil / a checkpoint is absent. See `docs/EVAL_TAMIL.md`.
- ✅ 📦 **Generation / inference** (`AdhanSLM.generate` + `src/adhan_slm/inference.py`
  + `scripts/generate_slm.py`): temperature / top-k / top-p / repetition-penalty
  sampling, loading tokenizer + checkpoint — the path Phase 4/5 need to sample text,
  feed the sandhi probe, and demo the model.
- 📋 Per-word perplexity; agglutination stress set; code-switch robustness
- 📋 Human read-through rubric (fluency, grammaticality) on generations
- 📦 MLflow evaluation runs, comparison table vs a real distilgpt2-Tamil baseline

### Phase 5 — Instruct / chat alignment (light)  📋  (Week 6–8)
- 📋 Small native-Tamil instruction set (translate + author + templated tasks)
- 📋 SFT of `adhan-nano` → `adhan-nano-instruct`; optional DPO if data allows
- 📦 Chat template + inference demo (reuse `scripts/run_model.py` pattern)

### Phase 6 — Compress & ship light  📋  (Week 7–9)  🚀 **Launch**
- 📋 Export to ONNX/GGUF/TFLite; INT8 + INT4 (existing `scripts/quantize_model.py`)
- 📋 Benchmark on **Raspberry Pi 5 / mid-tier Android** (latency, RAM, tok/s)
- 📋 Package: model card, license, `tokenizer.json`, minimal runner, HF Hub release
- 🎯 **≤ 30 MB INT4 nano, > 10 tok/s on-device, offline** → public v0.1 launch
- 📦 `adhan-nano-instruct-int4` release + announcement

### Phase 7 — Scale up  📋  (post-launch)
- 📋 Pretrain `adhan-tiny` / `adhan-mini` on TPU/multi-GPU (pjit sharding)
- 📋 Longer context, retrieval option, domain packs (news, literature, code-mix)

---

## 4. First launch definition of done (`adhan-nano` v0.1)

**Quality bar: kid-level Tamil** — basic reading, speaking, comprehension (like a 5–7 year old Tamil speaker).

1. Swaram tokenizer v1 frozen, fertility < 1.15 tok/akshara, lossless round-trip.
2. `adhan-nano` pretrained; per-akshara val ppl beats distilgpt2-Tamil baseline.
3. Instruct variant produces **kid-level grammatical Tamil** on a 50-prompt set:
   - Simple sentences, familiar words (body, family, food, numbers, colors)
   - Basic questions & commands
   - Age-appropriate stories (Panchatantra, folk tales)
4. INT4 build ≤ 30 MB, runs offline on Raspberry Pi 5 at > 10 tok/s.
5. Every run reproducible from MLflow (params + data version + code SHA).
6. **Model published via yazhi-api** (private `yazhi-lem/yazhi-api` repo):
   - REST endpoint for inference
   - Model card + license
   - Client library (Python SDK)
7. Demonstrate: read simple Tamil paragraph, answer basic questions, generate 2–3 sentences.

## 5. Risks & mitigations
| Risk | Mitigation |
|---|---|
| Corpus too small for from-scratch | Start at nano scale; curriculum; augment with morphological expansion |
| Morphological segmenter errors | Keep akshara fallback layer lossless; segmentation is additive signal only |
| JAX/TPU access | nano trains on a single 16–24 GB GPU; pjit only needed for mini |
| Scope creep vs "soon" | Ship nano first (Phases 0–6); tiny/mini deferred to Phase 7 |
| Reproducibility drift | MLflow logs code SHA + data version + config for every run |

## 6. Tokenizer family (Swaram + Aksharam + future)
| Tokenizer | Script | Target | Status | Purpose |
|---|---|---|---|---|
| **Swaram** | Tamil (தமிழ்) | Dravidian family (Tamil → Kannada/Telugu/Malayalam) | ✅ Phase 0 | akshara-native Dravidian tokenizer |
| **Aksharam** | Hindi/Devanagari | Indic script family (Hindi → Marathi/Bengali) | ✅ Phase 0 | matra/conjunct-native Indic tokenizer |
| Future | Others | … | 📋 | extend to other scripts (Urdu, Gujarati, etc.) |

Both live in a **shared tokenizer library** (`SwaramTokenizer` + `AksharamTokenizer`
share the merge/encode/decode machinery via `_segment`/`_base_inventory` hooks), with
a common **JAX-accelerated batch-encoding** fast path. Language-specific tuning only in
the script/morphology layer. Adhan builds on the **Yazh foundation** — see
[`docs/YAZH_FOUNDATION.md`](docs/YAZH_FOUNDATION.md).

## 7. Repository map (new)
```
src/adhan_slm/
├── tokenizer/   swaram_tokenizer.py  (+ tests)   # akshara + morpheme tokens (Dravidian prototype)
├── model/       transformer.py                    # Flax SLM (nano/tiny/mini) + generate()
├── data/        corpus.py, packing.py, loader.py  # corpus → packed shards → batches (+ tests)
├── training/    train_jax.py, mlflow_utils.py     # JAX loop, val, Orbax ckpt, MLflow
├── eval/        morphology.py, kid_level_prompts.py, ngram_baseline.py,
│                perplexity.py, run_eval.py         # probes + one-shot eval harness
├── external/    open_tamil_bridge.py               # base Tamil-NLP layer (open-tamil)
├── inference.py                                    # load tokenizer + checkpoint → generate
└── configs/     adhan_slm_tiny.yaml               # model+train+data+ckpt config
scripts/prepare_slm_corpus.py    # freeze tokenizer + pack shards + datasheet
scripts/generate_slm.py          # sample text from a trained checkpoint
requirements-jax.txt
docs/ARCHITECTURE_SWARAM_SLM.md, docs/EVAL_TAMIL.md
```

**Deployment (Phase 6+):**
- Model checkpoint → `yazhi-api` (private repo: `yazhi-lem/yazhi-api`)
- REST endpoint: `/models/adhan-nano/infer`
- Python SDK: `from yazhi_api import AdhanClient`

## 8. Getting started
```bash
python3 -m venv .venv-jax && source .venv-jax/bin/activate
pip install -r requirements-jax.txt

# Try the swaram tokenizer (no deps needed)
python -m adhan_slm.tokenizer.swaram_tokenizer "படித்துக்கொண்டிருந்தேன்"

# End-to-end data pipeline unit tests (pure-python, no JAX needed)
python -m adhan_slm.data.test_data_pipeline

# 1. Freeze the tokenizer + pack a corpus into train/val shards (+ datasheet)
python scripts/prepare_slm_corpus.py \
    --corpus data/raw/tamil/ --out data/final/tamil_slm \
    --vocab-size 12000 --seq-len 1024

# 2. Smoke-train (synthetic) then a real run on the packed shards
python -m adhan_slm.training.train_jax --config src/adhan_slm/configs/adhan_slm_tiny.yaml --smoke
python -m adhan_slm.training.train_jax --config src/adhan_slm/configs/adhan_slm_tiny.yaml
mlflow ui   # inspect runs at http://localhost:5000

# 3. Generate from a checkpoint, and run the full eval harness
python scripts/generate_slm.py --tokenizer-dir data/final/tamil_slm \
    --config src/adhan_slm/configs/adhan_slm_tiny.yaml \
    --checkpoint checkpoints/adhan-tiny --prompt "சொல், உனக்கு பிடித்த உணவு என்ன?"
python -m adhan_slm.eval.run_eval --tokenizer-dir data/final/tamil_slm \
    --eval-text data/final/tamil_slm --config src/adhan_slm/configs/adhan_slm_tiny.yaml \
    --checkpoint checkpoints/adhan-tiny --out docs/eval_report.json

# Once adhan-nano ships, test kid-level fluency
from yazhi_api import AdhanClient
client = AdhanClient(model="adhan-nano-instruct-int4")
text = client.generate("சொல், உனக்கு பிடித்த உணவு என்ன?")  # "Tell me, what food do you like?"
print(text)  # kid-level response expected
```

---
*Adhan — the first word, carried into the AI age.*  
*Making Tamil fluent at a kid's level, everywhere.*

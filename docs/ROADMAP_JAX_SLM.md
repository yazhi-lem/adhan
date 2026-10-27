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

JAX is the only training stack now (`src/adhan_slm/`, `requirements-jax.txt`).
The legacy PyTorch fine-tuning pipeline (Gemma LoRA, XLM-RoBERTa MLM,
sangam_gpt) has been removed; `src/data_scraper/` remains for corpus building
(Phase 2 data collection) only.

---

## 3. Phased roadmap

Legend: ✅ done · 🚧 in progress · 📋 planned. Live status is in `factory/CURRENT.md`; the dated plan is in
`factory/RUNWAY.md`. Each phase below keeps its heading so that code and docs can cite it.

### Phase A — Foundation & CI/CD ✅
Test, lint and type-check workflows; `pyproject.toml` + `pytest.ini` + shared fixtures; structured logging
(`core/logging.py`, JSON with `ADHAN_JSON_LOGS`) wired through the pipeline, with command *output* kept on stdout;
MLflow logs the runtime, effective batch, resume step, ETA, and config + datasheet artifacts (the §5 reproducibility
contract); deprecated scripts removed.

### Phase B — Robustness & Observability 🚧
Done: exception hierarchy, YAML config validation, throughput/latency/resource monitoring. **Open: full type hints
with zero mypy errors in CI.**

### Phase C — Deployment & Serving ✅
FastAPI `/generate`, `/tokenize`, `/decode`, `/health` (bounded input; 503 with no model loaded); Dockerfile + compose;
`docs/serving/DEPLOYMENT.md`; E2E CPU training and API integration tests. Open: a checkpoint → inference round trip from a
separate process; validation against a real trained checkpoint.

### Phase D — Roadmap Completion 📋
Close Phases 2–7 below.

### Phase 0 — Foundation & scaffolding ✅
`src/adhan_slm/` (tokenizer, model, training, configs, eval); Swaram and Aksharam tokenizers, tested; JAX batch
encoding; Flax nano/tiny/mini; MLflow-wired training loop; the Yazh foundation as a base requirement.

### Phase 1 — Tokenizer to production ✅
open-tamil as the Tamil-NLP layer (reference segmenter, stemmer, sandhi, encodings, lexicons;
`external/open_tamil_bridge.py`); `prepare_slm_corpus.py` freezes vocab + merges and writes a datasheet; Rust v2/v3
tokenizer with byte fallback, 100% lossless. 🎯 Fertility < 1.15 tok/akshara: met. Open: a morphology-guard decision
(see `factory/CURRENT.md`).

### Phase 2 — Corpus at pretraining scale 🚧
Pure-python data pipeline (`src/adhan_slm/data/`: corpus, packing, loader); streaming expansion with LSH dedup;
~610M swaram tokens across v1 + v2 (target 300M–1B clean, stretch 1.5–2B). Open: the cross-document packing fix,
a second near-dup pass, PII scrub, and a frozen `adhan-corpus-v1` with its data card.

### Phase 3 — Pretrain `adhan-nano` 🚧
Done: fused attention dispatch, donated train-step buffers, batched host sync; real-shard iterator; in-loop
validation + Orbax checkpoints with best-by-val retention and exact resume; CPU as a first-class path (bf16 ~1.5×
fp32, fp16 rejected on CPU, gradient accumulation); overfit-a-batch gate; Stage 1 + Stage 1b runs
(`docs/training/CONTINUED_PRETRAIN.md`). 🎯 Beat the classical floor and a distilgpt2-Tamil baseline per akshara.

### Phase 4 — Evaluation & Tamil-specific probes ✅
Morphology boundary agreement + sandhi rate (`eval/morphology.py`), 50 kid-level prompts, the classical unigram floor,
model perplexity, and one `run_eval` JSON report that degrades gracefully (`docs/behaviour/EVAL_TAMIL.md`); sampling and
inference (`inference.py`, `scripts/generate_slm.py`). Open: per-word perplexity, an agglutination stress set, a
human read-through rubric. Capability suites and ship bars: `docs/behaviour/CAPABILITIES.md` §6.

### Phase 5 — Instruct / chat alignment (light) 📋
SFT with the frozen response protocol (`docs/behaviour/RESPONSE_PROTOCOL.md`): Sangam + grounded QA + conversation + reasoning
+ refusal. DPO only with a verifier.

### Phase 6 — Compress & ship light 📋 🚀 Launch
Rework export for JAX checkpoints (SafeTensors/HF, ONNX), INT8 then INT4, benchmark on the reference phone /
Raspberry Pi 5, then the model card + HF release. 🎯 ≤ 30 MB INT4 nano, > 10 tok/s, offline.

### Phase 7 — Scale up 📋 (post-launch)
`adhan-tiny` / `adhan-mini` with FSDP; longer context; retrieval; domain packs. Mini only once ≥ 1B verified Tamil
tokens exist.

---

## 4. First launch definition of done (`adhan-nano` v0.1)

**Quality bar: kid-level Tamil**: basic reading, speaking and comprehension, like a 5–7-year-old Tamil speaker.

1. Swaram tokenizer frozen, fertility < 1.15 tok/akshara, lossless round trip.
2. `adhan-nano` pretrained; per-akshara val ppl beats the classical floor and a distilgpt2-Tamil baseline.
3. The instruct variant produces kid-level grammatical Tamil on the 50-prompt set (simple sentences, familiar words,
   basic questions, age-appropriate stories).
4. An INT4 build of ≤ 30 MB runs offline on a Raspberry Pi 5 at > 10 tok/s.
5. Every run is reproducible from MLflow (params + data version + code SHA).
6. Served through yazhi-api (REST endpoint, model card + licence, Python SDK).
7. Demo: read a simple Tamil paragraph, answer basic questions, generate 2–3 sentences.

## 5. Risks & mitigations

| Risk | Mitigation |
|---|---|
| Corpus too small for from-scratch | Start at nano scale; curriculum; morphological augmentation |
| Morphological segmenter errors | The akshara layer stays lossless; segmentation is additive signal only |
| GPU/TPU access | nano trains on CPU or a single T4; sharding is only needed for mini |
| Scope creep vs "soon" | Ship nano first (Phases 0–6); tiny/mini wait for Phase 7 |
| Reproducibility drift | MLflow logs code SHA + data version + config for every run |

## 6. Tokenizer family

| Tokenizer | Script | Family | Status |
|---|---|---|---|
| **Swaram** | Tamil | Dravidian (→ Kannada/Telugu/Malayalam) | ✅ production (Rust, 32k, byte fallback) |
| **Aksharam** | Devanagari | Indic (→ Marathi/Bengali) | ✅ prototype |

Both share one merge/encode/decode library, with language-specific logic only in the script/morphology layer.

## 7. Where things are

Package layout: [`src/adhan_slm/README.md`](../src/adhan_slm/README.md). Commands: [`docs/DEV.md`](DEV.md) and
[`docs/training/CPU_TRAINING.md`](training/CPU_TRAINING.md). Foundation: [`docs/architecture/YAZH_FOUNDATION.md`](architecture/YAZH_FOUNDATION.md).

---
*Adhan — the first word, carried into the AI age.*

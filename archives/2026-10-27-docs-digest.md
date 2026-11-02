# Adhan docs digest, 27 Oct 2026

A snapshot of this repo's documentation on 27 Oct 2026: what Adhan is, a map of the docs that remain, where they
disagree, and compressed copies of the 20 docs retired that day. It is never edited after that date. Live state is in
[`../factory/`](../factory).

## 1. Adhan in brief

A from-scratch, Tamil-first SLM (JAX). Its Swaram tokenizer removes the Tamil token tax (GPT-4 spends **4.21×** the
tokens); adhan-nano (14.5M) just beats the unigram floor; data, not compute, is the constraint (~610M tokens, ~164M
pure Tamil, no SFT data yet). Next: evaluate Stage 1b, fix packing, SFT v0, then Kutty v1 on 26 Dec.

## 2. Doc map (what remains, by theme)

| Theme | Docs |
|---|---|
| **Start here** | `README.md` (CLI, install) · `docs/DEV.md` (command sequence) · `docs/ROADMAP_JAX_SLM.md` (thesis, tiers, phases) · `factory/` (live plan) |
| **Tokenizer** | `docs/architecture/ARCHITECTURE_SWARAM_SLM.md` (design) · `docs/tokenizer/WHITEPAPER_SWARAM.md` (publishable results) · `docs/tokenizer/TOKEN_TAX_BENCHMARK_V3.md` (**canonical tax table**) · `docs/tokenizer/TOKENIZER_BENCHMARK_V2.md` / `_V3.md` (fertility, morphology agreement) |
| **Corpus** | `docs/corpus/CORPUS_EXPANSION.md` (v1 pipeline) · `docs/corpus/CORPUS_EXPANSION_V2.md` (v2 sources) · `docs/corpus/DIGITAL_LIBRARIES.md` (TVA, Wikisource) · `docs/corpus/CORPUS_HEALTH.md` · `docs/corpus/DATA_WEIGHTING.md` (purity, mixture) · `docs/corpus/SANGAM_INGEST.md` (one doc per song × layer) · `src/data_scraper/README.md` |
| **Training** | `docs/training/CONTINUED_PRETRAIN.md` (Stage 1b, gates G1–G5, eval protocol) · `docs/training/CPU_TRAINING.md` · `docs/training/COLAB_T4.md` · `docs/architecture/TINAI_CONDITIONING.md` |
| **Behaviour & eval** | `docs/behaviour/CAPABILITIES.md` (**the capability plan and ship bars**) · `docs/behaviour/RESPONSE_PROTOCOL.md` (frozen v0 wire format) · `docs/behaviour/REASONING_PROTOCOL.md` · `docs/behaviour/EVAL_TAMIL.md` · `docs/behaviour/ARAM_HARNESS.md` |
| **Serving** | `docs/serving/DEPLOYMENT.md` (FastAPI, Docker, K8s, yazhi-api) |
| **Strategy** | `docs/architecture/YAZH_FOUNDATION.md` · `docs/strategy/MULTILINGUAL_SCOPE.md` · `lang/README.md` · `docs/strategy/ADHAN_STORY.md` · `src/adhan_slm/README.md` |

## 3. Where docs disagree

| # | Finding | Fix |
|---|---|---|
| D1 | Fertility is published with **seven values** (0.304–0.554) taken from different samples and artifacts | One generated canonical table (`factory/NEXT.md` N3–N4) |
| D2 | Morphology agreement is quoted as 37.5% / 24.2% / 13.6% / 5.6%; gate G3 (≥ 30%) fails on the shipped tokenizer | Same table; decide on the guard after Stage 1b |
| D3 | `src/adhan_slm/README.md` listed a `rag/` package and `docs/RAG_YAZH.md`, neither of which exists | Row removed 27 Oct |
| D4 | The ONNX/quantize scripts target the removed HF pipeline | Treat export as not working; it is on Kutty's critical path |
| D5 | Kutty planning assumes fine-tuning an open base; the roadmap's thesis is from scratch | Decision `factory/NEXT.md` N2 |
| D6 | yazhi.dev calls Adhan "a 7B-parameter model for 22+ Indian languages" | Correct it before 31 Oct |
| D7 | `CAPABILITIES.md` cites `FOUNDER.md`, which does not exist (a dead `TRAINING_GUIDE.md` link and the README's broken fence were fixed on 27 Oct) | Drop the reference |
| D8 | No Stage 1b eval is committed, though the run was due to finish 25 Oct | `factory/NEXT.md` N1 |

## 4. Compressed copies of retired docs

**Status & planning, superseded by `factory/`**

- **`CURRENT.md` (architecture & production readiness).**
  - *Do not break:* Swaram tokenization; lazy JAX imports (the tokenizer works without JAX); tiṇai FiLM zero-init;
    GQA back-compat (`n_kv_heads=0`); guarded fused attention with an XLA fallback; self-describing checkpoints;
    config-hash resume safety.
  - *Phase 1 shipped:* data-parallel sharding, a persistent XLA cache (suite ~90 s), `MLFLOW_TRACKING_URI`, HF
    revision pinning (`scripts/_hf_pins.py`), a fixed-shape KV cache, input bounds (1–16,384 chars).
  - *Still open:* FSDP for mini, deterministic shuffle + checksum manifest, decode warm-up, a loss-scaling policy.
  - *Code-review checklist:* tests added; integration < 5 min; no `sys.path.insert`; no hard-coded paths; MLflow schema
    checked; tokenizer round-trip test if the tokenizer changed; Aram run if model behaviour changed; perf regression
    < 10%.
  - *Release:* release branch → full suite → Aram with 0 CRITICAL → benchmarks (≤ 10% regression) → semver tag →
    image → staging smoke → 10% traffic → 24 h → 100%.
  - *MLOps target:* Postgres + object-store MLflow, GCS/S3 checkpoints, a model registry, batched GPU/TPU serving,
    Prometheus, latency/error alerting, shadow A/B, automated rollback.
- **`NEXT_ACTION.md`.** The Oct pilot was tokenizer validation + a nano baseline. The Dec launch was 3-stage annealing
  (SCERT → synthetic Q&A → Sangam gold), SafeTensors/HF export for vLLM, INT8/INT4 ONNX for `yazh-unity` and
  `illakiya`, and an eval leaderboard. Data tiers: Gold = Open-Sangam + SCERT, Silver = Wikipedia + Project Madurai,
  Bronze = IndicCorp/Sangraha. Quality levers: MinHash LSH, Tamil ratio ≥ 0.80, NFC + pulli normalisation, a sandhi
  gate. *Stale:* it said "12k vocab".
- **`PROGRESS.md` (17 Aug).** Reference box: Ryzen 5 5600X, Quadro, 16 GB, memmap uint16 shards. Tier mix 35/30/35.
  Described the PyTorch `train_efficient.py` path; its ONNX export into `yazh-unity` no longer works with JAX models.
- **`docs/COMPLETION_TRACKER.md`, `docs/PHASE_A_TRACKER.md`.** Phase A (CI, structured logging, `pyproject.toml`,
  dead-code removal) closed 30 Jul. Phase B: full type hints never started. Phase C: serving, Docker and integration
  tests done. Command *output* deliberately stays on stdout.
- **`docs/ADHAN_NANO_PLAN.md`.**
  - *Corrections it recorded:* the pre-flight ETA was 1.6× optimistic (now measured after the first flush); the CPU
    profile is not the shipping profile (T4 is: 9L GQA 8q/2kv, 16.74M params); `dropout` was dead config before the
    GQA change.
  - *Param math:* with a 32k tied vocab, embeddings are 57% of the model.
  - *Post-training:* scope nano to completion, correction and curriculum tasks, and refuse open chat.
  - *Packaging estimate:* fp32 ~58 MB, int8 ONNX ~15 MB, tokenizer 2.8 MB.
  - *Production rules:* publish measured p50/p95; `/health` reports model and shard hashes; nightly eval alerts on a
    > 5% bits/akshara regression; canary → beta → public; checkpoints pinned by step; model card with code SHA,
    corpus and tokenizer hashes.
  - *Honest statement:* a Tamil text engine, not an assistant.
- **`docs/PHASE2_CORPUS.md`.** Target 300M–1B tokens. Pipeline: ingest → dedup (SHA-256 + Jaccard > 0.85) → filter
  (Tamil ≥ 70%, PII) → validate (fertility < 1.15) → pack. Source mix: ecosystem 5–15%, conversational 15–30%,
  news 15–30%, education + literature 30–40%. Its O(n²) scripts were replaced by `expand_tamil_corpus.py`.
- **`docs/CI_WORKFLOW_CLEANUP.md`.** Applied: install `-e ".[dev]"` everywhere; bandit non-blocking via
  `continue-on-error`. Proposed: one `ci.yml` and `setup-python@v5`. Check the required-check names first.

**Tokenizer, superseded by the whitepaper and the V2/V3 benchmarks**

- **`docs/SWARAM_AKSHARAM_TOKENIZERS.md`.** Two-layer lossless design shared by Swaram (Tamil, 247-akshara closed
  base) and Aksharam (Devanagari: halant chaining, nukta, anusvara, visarga, chandrabindu). `▁` word marker; Layer B
  morpheme BPE; bijective round trip. Its comparison claims (12k vocab, < 6 MB embeddings, 3.2–4.5× throughput) predate
  the 32k Rust tokenizer.
- **`docs/TOKEN_TAX_BENCHMARK.md` (v2).** GPT-4 4.36×, GPT-4o 1.21×, bytes 8.89× on 273 docs (swaram-v2 fertility
  0.543, 185 words per 512 tokens). Superseded by V3 (4.21× on 355 docs). The script still writes this path by default
  (`factory/NEXT.md` N3).
- **`docs/SWARAM_BENCHMARK.md`** (generated by `scripts/benchmark_tokenizers.py`). Thirukkural, 1,330 docs, 3k vocab:
  Swaram fertility 0.606 vs byte BPE 1.388 vs SentencePiece 0.450. Tamil script stress test (20 cases), NFC/byte-exact:
  Swaram 18/17, byte BPE 20/20, SentencePiece 8/7. The v1 Swaram `<unk>` failures were fixed by the byte fallback.

**Training data & behaviour, superseded by `CAPABILITIES.md` + `RESPONSE_PROTOCOL.md`**

- **`docs/TEACHING_ADHAN_TAMIL.md`.**
  - *Stages:* pretrain (clean, diverse, no English mixed into Tamil text; code-switch is a separate 5% slice) → SFT
    with the response protocol → alignment (DPO + Tamil constitutional critique).
  - *A good example is:* grammatical Tamil, culturally grounded, with shown reasoning.
  - *Task families:* sandhi/morphology, classical QA with citations, modern QA, translation, creative style,
    explanation, dialogue, code-switch awareness, refusal, tiṇai classification.
  - *Grounded generation:* build answers from source passages to avoid hallucination.

**Ops & expansion, deferred until after launch**

- **`docs/MLOPS_SRE_RUNBOOK.md`.**
  - *Tiers:* P0 training controller (99.5%) and checkpoint storage (99.99% durability, < 5 s save); P1 inference
    (99.9%, p99 < 200 ms/token) and MLflow; P2 ingestion cron (95%/day).
  - *Alerts:* loss spike > 3× median, val ppl +20%, tok/s < 50%, checkpoint save > 30 s, p99 > 500 ms, 5xx > 1%.
  - *Health states:* `loading` → 503, `ok` → 200, `error` → 503 + page.
  - *Rules:* run `mlflow db upgrade` before training; resume only when the config hash matches.
- **`docs/MULTILINGUAL_EXTENSION_GUIDE.md`.**
  - *Adding a script:* define its Unicode geometry, Layer A segmenter, inventory, tokenizer registration and
    benchmark. Bootstrap Kannada merges from Tamil.
  - *Per-script traps:* Devanagari conjuncts and ZWNJ half-forms; Bengali hasant ligatures (decompose them);
    Telugu/Malayalam pre-base matras (grapheme `\X` first pass).
  - *Vocabulary strategy:* start with a shared Dravidian vocab (24k), then a Devanagari cluster. Upsample
    low-resource languages.
  - *Targets:* fertility < 1.15 for Dravidian, < 1.25 for Devanagari.

**Obsolete (described removed code)**

- **`src/data_scraper/export/README.md`, `processing/README.md`.** Usage notes for `export_unified_hf.py` and
  `build_unified_corpus.py` (modern/balanced/rebalanced strategies), plus "removed" notices for four deleted
  scripts. They are now covered by the package README and each script's `--help`.

- **`docs/PERFORMANCE.md`.** HF/PyTorch tips for the removed fine-tuning pipeline.
- **`docs/SECURITY_HARDENING.md`, `docs/SECURITY_SUMMARY.md`.** Controls for removed scripts and a Feb 2026 CodeQL scan
  of deleted files. *Still true:* bandit blocks merges on High severity; 18 Medium findings (mostly unpinned HF
  revisions) are tracked, and `_hf_pins.py` is the fix path.
- **`docs/archive_TVA_BOK_0005884_page_n39.md`.** One OCR page from J. Vacek, *A Tamil Reader: Introducing Sangam
  Literature* (1989). It was source material, not documentation; the full text is item `TVA_BOK_0005884` in the
  Internet Archive's Tamil Virtual Academy collection.

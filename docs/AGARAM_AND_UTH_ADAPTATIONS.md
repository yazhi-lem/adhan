# Adhan Adaptations from Agaram 2.5B and "Under the Hood"

**Status**: In progress — modules created, integration points documented.

This document records what was learned from studying two repositories and how
the lessons were adapted into the Adhan SLM codebase.

---

## Source Repositories

| Repository | Author | Purpose | Licence |
|------------|--------|---------|---------|
| [Murai-Labs/Agaram2.5b](https://github.com/Murai-Labs/Agaram2.5b) | Ramchand Kumaresan, Murai Labs | Production 2.5B Tamil–English LLM trained from scratch. Code companion to arXiv:2610.18420. | Apache-2.0 |
| [mechramc/Under-the-hood](https://github.com/mechramc/Under-the-hood) | Ramchand Kumaresan | 36-project educational book companion: build every layer of a Transformer from scratch. | MIT |

---

## 1. Data Curation & Contamination (Agaram + Under-the-Hood Proj. 10)

### What we learned

Agaram 2.5B used **MinHash LSH** to deduplicate 87.7M → 61.3M documents, then
removed **42,001 documents** overlapping 20 benchmark/eval sets.  The paper
discloses exact overlap rates and clean-subset scores.

Key insight: **Contamination is not just exact-match**. Agaram filters by:
- Exact character match
- Word 8-gram containment ratio
- Character 5-gram Jaccard similarity

Under the Hood Project 10 emphasises: *"what makes good pretraining data"* —
clean, diverse, non-contaminated.

### What was adapted

#### `src/adhan_slm/data/minhash_dedup.py` — MinHash deduplication
- Configurable `n_hashes`, `shingle_size`, `threshold`
- LSH banding for sub-quadratic candidate generation
- CLI: `python -m adhan_slm.data.minhash_dedup corpus.jsonl --output clean.jsonl`

#### `src/adhan_slm/data/contamination.py` — Benchmark contamination detection
- Three detection methods: exact match, word n-gram overlap, char n-gram Jaccard
- `BenchmarkIndex.from_files()` builds an index from benchmark JSONL files
- `detect_contamination()` filters corpus and produces a `ContaminationReport`
- CLI: `python -m adhan_slm.data.contamination scan --index benchmarks/ --corpus raw.jsonl --output clean.jsonl`

### Integration point

Call from `scripts/prepare_slm_corpus.py` before packing shards::

    # After scraping/cleaning, before prepare_slm_corpus packs train.bin
    from adhan_slm.data.minhash_dedup import deduplicate_records, DedupConfig
    from adhan_slm.data.contamination import BenchmarkIndex, detect_contamination

    records = load_corpus_records(...)
    unique, clusters = deduplicate_records(records, DedupConfig(n_hashes=128))
    index = BenchmarkIndex.from_files(glob("data/benchmarks/*.jsonl"))
    clean, report = detect_contamination(unique, index)
    write_jsonl(clean, "data/deduped_clean.jsonl")

---

## 2. Training Debugging (Under-the-Hood Project 11)

### What we learned

Project 11 teaches *per-layer* diagnostics:
- **Gradient norms per layer**: detect which layer spikes before NaN
- **Activation histograms**: catch saturation or dead ReLU-equivalents
- **Weight-update ratios**: ``||Δw|| / ||w||``, the "health" of optimisation
- **Gradient clipping**: `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)`
- **AMP overflow detection**: scaler scale drops → overflow event

Agaram 2.5B trains at 524K tokens/step with no restarts across 8 H100s — stable
training requires these diagnostics.

### What was adapted

#### `src/adhan_slm/training/debug_diagnostics.py` — JAX/Flax diagnostics
- `TrainingDiagnostics` class: captures grad norms, activation stats, AMP overflow
- `_group_grad_norms()`: groups by top-level param key (`block_0`, `tok_emb`, …)
- `detect_spikes()`: rolling-window z-score spike detection (>3σ)
- `detect_nans()`: scans latest activation stats for Inf/NaN
- JSON log files written to `log_dir/diag_step_00000001.json`

### Integration point

Extend `train_jax.py` to instantiate diagnostics and call `maybe_log()`:

    diag = TrainingDiagnostics(model, log_every=log_every, log_dir="logs/diagnostics")
    for step in range(updates):
        state, loss, grads = train_step(...)
        diag.maybe_log(state, step, float(loss), lr=float(sched(step)), grads=grads)
        spikes = diag.detect_spikes()
        if spikes:
            logger.warning("Gradient spikes detected: %s", spikes)

---

## 3. KV Cache & GQA (Under-the-Hood Projects 13 & 15)

### What we learned

- **Project 13 (KV Cache)**: Naïve autoregressive generation re-computes K/V for
  the full prefix at every token.  A per-layer cache of fixed-size buffers
  written at `offset` via `dynamic_update_slice` cuts this to O(1) per step.
- **Project 15 (Grouped Query Attention)**: Sharing K/V heads across query-head
  groups reduces KV memory — critical for long context on limited VRAM.

### What was adapted

Adhan's `transformer.py` **already implements both** (credit to prior Adhan work):
- KV cache via `dynamic_update_slice` in fixed-size `[B, S_max, kvh, dh]` buffers
- GQA via `n_kv_heads` config parameter with broadcast-to-expand

**New additions** (from Under-the-Hood):
- Sliding-window KV cache truncation (future): ``if cache_len > window: k_cache = k_cache[:, :, -window:, :]``
- Document boundary masking (future): prevent attention across document boundaries
  in packed training (Agaram does this; Adhan's shard packer currently does not).

---

## 4. Crisis Safety (Agaram 2.5B)

### What we learned

Agaram's `serving/agaram_chat.py` includes a **wrapper-level** crisis footer:
- Detects distress phrases in Tamil, Tanglish, and English
- Prints Tele-MANAS 14416 / 112 resources
- Explicitly **not** a model capability — the model may fail to respond appropriately

### What was adapted

#### `src/adhan_slm/safety/crisis.py` — Crisis detection & footer
- `CRISIS_PHRASES`: 30+ phrases in Tamil script, Tanglish, and English
- `needs_footer(text)`: case-insensitive substring matching
- `CRISIS_FOOTER`: bilingual Tamil/English resource line

#### `src/adhan_slm/chat_cli.py` — Chat CLI with safety wrapper
- Adapts Agaram's CLI pattern to Adhan's JAX/Flax + Swaram stack
- Uses Tamil role protocol (`பயனர்:`, `உதவியாளர்:`) instead of ChatML
- Calls `needs_footer()` after every user input
- Supports greedy (`temperature=0`) and sampling modes
- Shipped as the `adhan-chat` console script; `scripts/adhan_chat.py` is a thin shim so it
  also runs from a checkout

---

## 5. GGUF Export & Token Parity (Agaram 2.5B)

### What we learned

Agaram provides:
- `convert_agaram.py`: wraps `convert_hf_to_gguf.py`, sets pre-tokenizer to `qwen35`
- `gguf_parity.py`: compares token IDs between HF tokenizer and GGUF on 961 texts
- `gguf_parity.py` also tests greedy replies: f16 matches on 5/8, Q8_0 on 5/8, Q4_K_M on 0/8

Key insight: **Token parity is not automatic**. Different pre-tokenizer regex
patterns (ZWJ/ZWNJ inclusion) cause divergence. Adhan's Swaram tokenizer will
need its own parity verification.

### What was adapted

#### `scripts/export_gguf.py` — JAX → HF → GGUF pipeline
- `export_jax_to_hf()`: documents the full parameter mapping from Adhan JAX → Llama HF
- Chat template written as Jinja (Tamil protocol)
- `convert_hf_to_gguf()`: wraps llama.cpp converter

**Status**: Weight-copy mapping is documented but not fully implemented —
requires careful GQA head splitting and transpose handling.

**Future**: Add `scripts/gguf_parity.py` that tests token IDs and greedy replies
against the JAX model, analogous to Agaram's parity checker.

---

## 6. Evaluation Methodology (Agaram + Under-the-Hood Proj. 22)

### What we learned

Agaram's evaluation practices:
- All benchmarks run with **lm-eval** on their own hardware, same tasks/settings
- **Sealed test sets**: 100 human-written prompts kept unseen until checkpoint frozen
- **LLM judge** (Kimi K3) for pairwise comparison, with bootstrap confidence intervals
- Disclosures: which eval items also appear in training data, how checkpoint chosen

Under the Hood Project 22: systematic evaluation — train/test splits, statistical
significance, human baselines.

### What was adapted

Adhan's existing `eval/run_eval.py` already covers Tamil-specific probes
(perplexity, morphology, fertility).  **New recommendations** (not yet coded):

1. **Adopt lm-eval harness** for standard benchmarks (FLORES, XCOPA, MILU)
2. **Sealed prompt set**: 100 human-written Tamil prompts for chat evaluation
3. **Pairwise LLM judge**: compare checkpoints with confidence intervals
4. **Clean-subset reporting**: report scores on the contamination-free subset

---

## 7. Tokenizer Architecture Insight

### Comparison

| Model | Tokenizer | Tokens / Tamil word | Approach |
|-------|-----------|---------------------|----------|
| Agaram 2.5B | Byte-level BPE (64k) | **1.55** | Learns byte patterns |
| Qwen2.5 | Byte-level BPE | 9.44 | Foreign multilingual base |
| **Adhan** | **Swaram akshara + BPE** | **<1.15** | Linguistic + learned |

Adhan's Swaram tokenizer is **more linguistically grounded** than Agaram's
pure byte-level BPE, but Agaram's approach achieves excellent practical
performance (1.55 tokens/word) without hand-crafted rules.

**Lesson**: Both approaches are valid. Adhan should keep Swaram for linguistic
fidelity and sandhi correctness, but Agaram's success proves that a well-trained
BPE on clean Tamil text is competitive.  The Swaram-BPE hybrid (Layer A + B)
is the best of both worlds.

---

## 8. Transparency & Disclosure (Agaram)

### What we learned

Agaram's README is a model of transparency:
- Full data provenance with licence terms
- Exact contamination overlap numbers
- How the release checkpoint was chosen (dev sets → sealed set)
- Frontier-model text disclosure (Codex user turns, Alpaca/Orca data)
- Cost disclosure (~$4,765 pretraining)

### What was adapted

Added to this document and recommended for Adhan release cards:
- Data source licences (Project Madurai, Open-Sangam, Wikipedia)
- Contamination analysis (using `minhash_dedup.py` + `contamination.py`)
- Evaluation split methodology
- Cost estimates (for future pretraining runs)

---

## 9. SFT Recipe (Agaram + Under-the-Hood Proj. 21)

### What we learned

Agaram SFT: **one run from base**, 222,662 records, 1 epoch, LR 1e-5, step 700 chosen
on dev sets.  Key design choices:
- Translation lanes (English→Tamil, Tamil→English) ≈ 20% of SFT tokens
- General chat + self-play + tool calls + identity + crisis replies
- Targeted sets aimed at failures from preview evaluation
- All reported chat results use **greedy decoding**

Under the Hood Project 21 covers LoRA, full SFT, conversational tuning.

### What was adapted

Adhan's existing `train_sft.py` + `build_sangam_sft.py` / `build_reasoning_sft.py`
covers SFT for the Sangam domain.  **Recommendations**:
- Add general chat SFT data (beyond Sangam) for broader Tamil competence
- Consider tool-call training (Adhan protocol supports `<tool_call>` syntax)
- Add identity-refusal and crisis-reply targeted sets
- Use greedy decoding for all reported eval

---

## File Summary

| File | Source Inspiration | Purpose |
|------|-------------------|---------|
| `src/adhan_slm/data/minhash_dedup.py` | Agaram + UTH Proj. 10 | MinHash LSH deduplication (`adhan-dedup`) |
| `src/adhan_slm/data/contamination.py` | Agaram + UTH Proj. 10 | Benchmark leak detection (`adhan-decontaminate`) |
| `src/adhan_slm/training/debug_diagnostics.py` | UTH Proj. 11 | Per-layer grad/activation diagnostics |
| `src/adhan_slm/safety/crisis.py` | Agaram `serving/agaram_chat.py` | Crisis detection & footer |
| `src/adhan_slm/chat_cli.py` | Agaram `serving/agaram_chat.py` | Chat CLI with safety wrapper (`adhan-chat`) |
| `scripts/adhan_chat.py` | — | Shim so the CLI runs from a checkout |
| `scripts/export_gguf.py` | Agaram `gguf/convert_agaram.py` | JAX → HF → GGUF export (`[export]` extra) |
| `scripts/build_package.sh` | — | Build gate: lint, tests, entry points, sdist+wheel |
| `docs/AGARAM_AND_UTH_ADAPTATIONS.md` | This document | Adaptation record |

---

## 10. Packaging / build changes

`pyproject.toml` gained, in `[project.scripts]`:

| Script | Module | Needs |
|---|---|---|
| `adhan` | `adhan_slm.cli:main` (unchanged) | core |
| `adhan-dedup` | `adhan_slm.data.minhash_dedup:_cli` | core |
| `adhan-decontaminate` | `adhan_slm.data.contamination:_cli` | core |
| `adhan-chat` | `adhan_slm.chat_cli:main` | `[jax]` at runtime |

and a new `[export]` extra (`torch`, `transformers`, `safetensors`, `huggingface_hub`) so the
GGUF path does not drag ~2 GB of torch into the training environment. `[all]` includes it.

`scripts/build_package.sh` is the build gate:

```bash
./scripts/build_package.sh          # black, isort, ruff, pytest, entry points, sdist+wheel
./scripts/build_package.sh --fast   # same minus pytest
```

It resolves every declared console script to a real callable and asserts the wheel
contains the new modules — a package whose entry points do not import fails the build
instead of failing for a user.

Verified 2026-11-05: `ruff check src tests scripts` clean, `pytest tests/ src/adhan_slm`
**320 passed / 2 skipped**, `adhan-dedup` and `adhan-decontaminate` both run end-to-end on a
3-document Tamil sample.

**Known caveat:** `python -m build` is not runnable in the current `.venv` (no `hatchling`
installed), so the sdist/wheel step of the script must be run where the build backend is
available (CI, or `pip install build hatchling`). Pre-existing `black` drift on files this
work did not touch (`packing_tests.py`, `gqa_tests.py`) comes from black-version
parenthesisation, not from these changes.

---

## Next Steps

Tracked in [`../factory/NEXT.md`](../factory/NEXT.md) as K11–K13, not committed this iteration.

| # | Work | Factory ID |
|---|------|-----------|
| 1 | Call dedup + leak filter from `scripts/prepare_slm_corpus.py`; record counts in `datasheet.json` | K11 |
| 2 | Set `train.diagnostics_every` on the next real GPU run and triage the first spike report | K12 |
| 3 | Implement the JAX→HF weight copy in `scripts/export_gguf.py` + a token-parity test | K13 |
| 4 | Sliding-window KV cache in `transformer.py` (UTH Proj. 13 step 8) | research |
| 5 | Document-boundary masking in the packed shard loader (Agaram packs with per-document masks) | research |

Done already: diagnostics wired into `train_jax`; entry points and build gate added; the
full suite is green (`pytest tests/ src/adhan_slm` → 320 passed / 2 skipped).

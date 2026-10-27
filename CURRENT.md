# Adhan SLM — Architecture, Roadmap & Production Readiness

> **Purpose**: Single source of truth for Adhan's current state, known gaps, and prioritized path to production. Updated after every significant milestone.

---

## 1. Current Architecture

```
Adhan SLM
├── Model: adhan-nano (~8.3M params, 256 d_model, 6L, 4H, 512 ctx)
├── Tokenizer: Swaram v2 (akshara + BPE, fertility < 1.15)
├── Training: JAX/Flax, single-device, Orbax checkpointing
├── Serving: FastAPI (async loading, rate limiting, timeouts)
├── Safety: Aram Harness (12 constitutional principles, 27 red-team prompts)
├── Reasoning: 4-stage Tamil thinking (சிந்தனை→ஆய்வு→பரிசீலனை→பதில்)
└── Corpus: 328M tokens (1.33M docs), Phase 2 targeting 1.5–2B
```

---

## 2. Strengths (Do Not Break)

| Decision | Why It Matters |
|----------|---------------|
| Swaram tokenization | Fertility < 1.15 vs. 3–6 for byte-level BPE. Publishable research. |
| Lazy JAX imports | Tokenizer works without JAX stack. Correct dependency isolation. |
| Tiṇai FiLM zero-init | Enabling conditioning starts from exact unconditioned checkpoint. |
| GQA backward compat | `n_kv_heads=0` preserves old checkpoint shapes. Forward-compatible. |
| Fused attention dispatch | Guarded `jax.nn.dot_product_attention` with XLA fallback. |
| Self-describing checkpoints | Metadata JSON with `model_config` + `tokenizer_dir`. |
| Config-hash resume safety | SHA256 prevents mixing incompatible training runs. |

---

## 3. Production Hardening Status

### 3.1 Distributed Training — ADDRESSED (data-parallel)

**Shipped**: `src/adhan_slm/training/sharding.py` provides a 1-D data-parallel
mesh. Enable with `train.n_devices: N` in the config (default 0 = all visible
devices, 1 = disabled). Each device holds a full param replica and processes
`global_batch / N` examples; the loss is wrapped in `jax.lax.pmean`, which also
all-reduces gradients. Single-device and sanity modes are byte-for-byte
unchanged (the helper degrades to a no-op).

**Still open**: FSDP/tensor-parallel for `adhan-mini` (params exceed one device
only at larger sizes); multi-host. Reference: `t5x`, `maxtext`.

### 3.2 Test Suite — ADDRESSED

**Shipped**: suite runs ~90s (was timing out >120s) thanks to a persistent XLA
compilation cache (`ADHAN_JAX_CACHE_DIR`, disable with `ADHAN_DISABLE_JAX_CACHE=1`).
Added: reasoning-mask correctness (incl. prompt containing `பதில்:`), KV-cache vs
full-forward equivalence, sharding helpers, tokenizer round-trip (inventory +
fuzzed + punctuation with byte fallback), checkpoint restore determinism and
resume-actually-updates-weights.

**Still open**: performance-regression gates in CI; Aram Harness automation;
property-based checks over the whole frozen corpus.

### 3.3 MLflow Backend — ADDRESSED (configurable)

**Shipped**: `MLFLOW_TRACKING_URI` env override (explicit arg still wins) and an
upfront SQLite schema check that logs an actionable `mlflow db upgrade` message
instead of failing mid-run.

**Remaining**: standardise on Postgres or a shared file store for multi-host;
add artifact-store config for GCS/S3.

### 3.4 Data Pipeline Determinism — ADDRESSED (revision pinning)

**Shipped**: `scripts/_hf_pins.py` centralises `load_dataset`. Pin any source
with `ADHAN_HF_REVISION_<org>_<name>=<sha>`; the revision used is recorded under
`hf_revisions` in `stats.json`. Both corpus scripts use it.

**Remaining**: deterministic document-level shuffle + checksum manifest so two
runs produce identical shard hashes.

### 3.5 Inference KV-Cache — ADDRESSED

**Shipped**: fixed-size KV buffers per layer with dynamic write + explicit
temporal mask. Shapes are constant across decode steps, so a compiled step is
reused (no per-token recompile). `generate()` uses it automatically for
non-Tiṇai models; a decode-step equivalence test asserts cached logits match a
full forward pass.

**Remaining**: startup warm-up of the decode program (first call pays compile
cost, then served from the persistent XLA cache).

### 3.6 Mixed Precision — PARTIAL

`dtype` is validated and cast at layer boundaries; bf16/fp32 both work. Still
to do: explicit loss-scaling policy and a documented fp16 path for older GPUs.

### 3.7 Serving Input Bounds — ADDRESSED

`AdhanRequest.text` is now bounded (`1..16384` chars) so a hostile payload
cannot OOM a worker before tokenization.

---

## 4. Production Roadmap

### Phase 1: Foundation Hardening — ✅ COMPLETE

| # | Task | Status | Where |
|---|------|--------|-------|
| 1 | Reasoning mask boundary bug | ✅ | `training/train_sft.py` |
| 2 | API input length validation | ✅ | `serving/api.py` |
| 3 | `jax.sharding` data-parallel | ✅ | `training/sharding.py` + `train_jax.py` |
| 4 | Integration test timeout | ✅ | XLA cache in `training/device.py` |
| 5 | MLflow backend flexibility | ✅ | `training/mlflow_utils.py` |
| 6 | KV-cache in inference | ✅ | `model/transformer.py` |
| 7 | HF dataset revision pinning | ✅ | `scripts/_hf_pins.py` |
| 8 | Tokenizer round-trip tests | ✅ | `tests/test_tokenizer_roundtrip.py` |
| 9 | Checkpoint resume round-trip | ✅ | `tests/integration/train_cpu_checkpoint_tests.py` |

Exit criteria met: full suite 96 passed / 1 skipped in ~90s; smoke and
overfit-a-batch gates still pass; single-device behaviour unchanged.

### Phase 2: Scale & Quality (Weeks 3–6) — 🚧 IN PROGRESS

| # | Task | Why | Status |
|---|------|-----|--------|
| 10 | Token-tax benchmark vs GPT-4/GPT-4o/byte | Evidence the tokenizer removes the Tamil tax | ✅ `docs/TOKEN_TAX_BENCHMARK.md` |
| 11 | Run Phase 2 corpus expansion (target 1.5–2B) | More data = better model | 🚧 running, resumable |
| 12 | Retrain tokenizer v3 on expanded corpus | ✅ `tamil_slm_v3`: vocab 31,964, fertility 0.506, lossless 100% |
| 13 | Validate v3 data path (shards → train) | ✅ 40-step run: loss 9.02 → 6.69, val_ppl 795 |
| 14 | Pretrain adhan-nano on expanded corpus | Baseline for downstream tasks | ⏳ pending (raise max_steps) |
| 15 | Run Aram Harness on pretrained checkpoint | Safety gate before any SFT | ⏳ pending |
| 16 | Implement FSDP for adhan-mini | Unlocks 110M param training | ⏳ pending |

Phase 2 progress (tokenizer v3):

| metric | v2 (old) | v3 (expanded) |
|---|---|---|
| vocab | 31,964 | 31,964 |
| fertility (token-tax sample) | 0.564 | **0.554** |
| tokens / Tamil word | 2.926 | **2.874** |
| Tamil words in 512-token ctx | 175 | **178** |
| round-trip | 100% | 100% |
| token tax vs GPT-4 (cl100k) | 4.36x | **4.21x** |
| token tax vs GPT-4o (o200k) | 1.21x | 1.21x |

Artifacts: `data/final/tamil_slm_v3` (tokenizer + packed shards,
338,758 train sequences x 512). Reports: `docs/TOKEN_TAX_BENCHMARK_V3.md`,
`docs/TOKENIZER_BENCHMARK_V3.md`. Config:
`src/adhan_slm/configs/adhan_slm_nano_cpu_v3.yaml`.

Open quality gap: morphology boundary agreement is ~5.6% for both v2 and v3 —
BPE merges cut through stem/suffix boundaries. `freeze_fast_tokenizer.py
--morphology-guard` is the intended fix (trades some compression for
linguistically aligned boundaries) and should be evaluated next.

Phase 2 progress (corpus):

| metric | value |
|---|---|
| v1 archive (baseline) | 1,334,126 docs · 1.06B chars · ~328M swaram tokens |
| v2 archive (new, deduped vs v1) | 538,381 docs · 900M chars · ~281M swaram tokens |
| combined | ~1.57M docs · ~1.96B chars · **~609M swaram tokens** |
| shards | 23 (data/raw/expanded_v2) |
| sources that landed | mc4 198,864 · indiccorp 197,082 · fineweb-2 42,150 · local classical · wikipedia · tamilvu 850 · wikisource 77 |
| sources unavailable | cc100 (script datasets no longer supported) · oscar (gated, needs HF_TOKEN) · news/books (0 accepted) |

Digitised libraries are now part of the pipeline: `scripts/ingest_tamilvu.py`
(Tamil Virtual Academy — Sangam, grammar, epics, dictionaries) and
`scripts/ingest_wikisource_ta.py` (Tamil Wikisource). See
[`docs/DIGITAL_LIBRARIES.md`](docs/DIGITAL_LIBRARIES.md). Their output is merged
through `data/raw/weblibs` and counted by `--recount-only`.

To reach the 1.5–2B target, raise limits on the sources that work
(`--indic-limit`, `--mc4-limit`, `--fineweb-limit`) and resume; `cc100`/`oscar`
need a converted mirror or an authenticated token. The ingest runner now
isolates per-source failures and always writes cumulative stats, so a gated
dataset can no longer abort the build.

Phase 2 kickoff notes:

- Expansion uses `scripts/expand_tamil_corpus_v2.py`, which is resumable via
  `state.json` and dedups cross-version against the v1 archive. Sources are
  revision-pinned (`ADHAN_HF_REVISION_*`).
- The token-tax result is the Phase 2 gate for the tokenizer: swaram must stay
  at **1.00x**; any retrain that raises fertility/tox is rejected.
- After expansion, retrain the tokenizer and re-run `benchmark_token_tax.py`;
  the comparison table in the doc is regenerated, not hand-edited.

### Phase 3: Alignment & Deployment (Weeks 7–12)

| # | Task | Why | Effort |
|---|------|-----|--------|
| 15 | SFT with reasoning protocol (50K examples) | Conversational capability | 3 days |
| 16 | DPO alignment with Tamil constitutional AI | Values enforcement | 1 week |
| 17 | INT8/INT4 quantization with calibration | On-device deployment | 1 week |
| 18 | ONNX export with verified correctness | Cross-platform inference | 2 days |
| 19 | Deploy API to Cloud Run with load testing | Production serving | 3 days |

---

## 5. Engineering Standards

### Code Review Checklist (enforce in PR template)
- [ ] Tests added for new functionality
- [ ] Integration tests pass locally (<5 min)
- [ ] No `sys.path.insert` in library code
- [ ] No hardcoded paths (config or env vars only)
- [ ] MLflow schema version checked
- [ ] Tokenizer round-trip test added if tokenizer changed
- [ ] Aram Harness run if model behavior changed
- [ ] Performance regression < 10%

### Release Process
1. Cut release branch from `main`
2. Run full test suite + integration tests
3. Run Aram Harness (zero CRITICAL failures)
4. Run performance benchmarks (no regression >10%)
5. Tag with semantic version (`v0.3.0`)
6. Build Docker image, push to registry
7. Deploy to staging, run smoke tests
8. Deploy to production with 10% traffic
9. Monitor for 24h, then 100% traffic

---

## 6. MLOps Requirements

| System | Current | Required |
|--------|---------|----------|
| Experiment tracking | MLflow SQLite | PostgreSQL + GCS artifacts |
| Checkpoint storage | Local disk | GCS/S3 with lifecycle policy |
| Model registry | None | MLflow Model Registry or custom |
| Serving | FastAPI single process | TPU/GPU batched with queue |
| Monitoring | stderr logs | Cloud Logging + Prometheus |
| Alerting | None | PagerDuty for latency/error rate |
| A/B testing | None | Shadow traffic for variants |
| Rollback | Manual | Automated on error rate spike |

---

## 7. Cross-Yazhi Integration Map

```
[adhan] Swaram Tokenizer ──► [illakiya] Trie Dictionary & IME
[adhan] ONNX Export ──────► [yazh-unity] On-Device Pet AI
[adhan] Corpus ───────────► [open-sangam] Pretraining Data
[blender-mcp] 3D Meshes ──► [yazh-unity] & [yazhi-dev]
[yazhi-skills] Guardrails ─► ALL Repositories
```

---

## 8. Key Metrics

| Metric | Current | Target |
|--------|---------|--------|
| Pretrain tok/s (CPU, nano) | ~2,000 | Baseline |
| Pretrain tok/s (TPU v4) | Not measured | ~50,000 |
| Inference latency (48 tok) | KV-cached; steady-state after one compile | <200ms |
| Fertility (tok/akshara) | 0.543 (swaram-v2) | Maintain <1.15 |
| Token tax vs GPT-4 (cl100k) | **1.00x vs 4.36x** | Keep swaram at 1.00x |
| Token tax vs GPT-4o (o200k) | **1.00x vs 1.21x** | Keep swaram at 1.00x |
| Token tax vs UTF-8 bytes | **1.00x vs 8.89x** | — |
| Test pass rate | 96 passed / 1 skipped (~86s) | CI green |
| Aram Harness score | Not run | >0.90 |

### Token tax (showcase)

Measured on the real archive with English-centric tokenizers as baselines —
see [`docs/TOKEN_TAX_BENCHMARK.md`](docs/TOKEN_TAX_BENCHMARK.md) and
`scripts/benchmark_token_tax.py`.

| tokenizer | tax vs swaram | words in 512 ctx |
|---|---|---|
| swaram-v2 (ours) | **1.00x** | 185 |
| GPT-4o (o200k) | 1.21x | 154 |
| GPT-4 (cl100k) | 4.36x | 42 |
| char (codepoint) | 3.35x | 55 |
| byte (UTF-8) | 8.89x | 21 |

The tax is paid three times over: training compute, API cost, and effective
context. The swaram tokenizer is what removes it.

---

## 9. Reading List for Team

1. **JAX/Flax**: `jax.sharding` docs, `flax.linen.spmd` examples
2. **Distributed Training**: T5X mesh setup, MaxText FSDP
3. **Inference Optimization**: KV-cache implementation in `transformers`
4. **Testing**: Hypothesis property testing, pytest-xdist
5. **MLOps**: Seldon Core or KServe for model serving
6. **Tamil NLP**: `open-tamil` docs, Project Madurai encoding standards

---

*"கற்றது கைமண்ணளவு, கல்லாதது உலகளவு" — What we know is a handful; what remains is the world.*

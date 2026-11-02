# CURRENT — adhan · 2026-10-27

> Reset every Monday. Facts only; each number cites its source. The template and rules are in [`README.md`](README.md).

## 1. Headline

**🟡 Amber.** The tokenizer is the strongest asset Yazhi has, and it is ready to publish: a measured **4.21×** token
tax for GPT-4 on Tamil. The model is a working plumbing milestone, not yet a product. adhan-nano barely beats the
unigram floor, there is no SFT data, and the Aram safety harness has never run. The next public date is the
**tokenizer benchmark campaign on 31 Oct (4 days)**. The date with the most risk is **Adhan Kutty v1 on 26 Dec
(60 days)**. Two things stand in its way: an open decision on what Kutty *is* (from scratch, or a fine-tune of an open
base), and an export/quantize path that does not work with the JAX model today.

## 2. Health

| Workstream | RAG | Fact (source) | Next checkpoint |
|---|---|---|---|
| Tokenizer (Swaram v3) | 🟢 | vocab 31,964 · 100% lossless · fertility 0.554 · tax vs GPT-4 4.21× (`docs/tokenizer/TOKEN_TAX_BENCHMARK_V3.md`, 355 docs) | Public benchmark + repro, 31 Oct |
| Tokenizer morphology (G3) | 🔴 | boundary agreement 5.6% (v3 sample) / 24.2% (Stage 1b sample); gate is ≥30%. The morph-guard variant reached 37.5% at +23% tokens (`TOKENIZER_BENCHMARK_V2.md`) | Decide whether v4 uses the guard |
| Corpus scale | 🟡 | ~609.7M swaram tokens (v1 328M + v2 281.7M), 1.87M docs. Target 1.5–2B, so ~35% of the way (Phase 2 totals, 27 Oct) | Raise IndicCorp / mC4 / FineWeb limits |
| Corpus purity | 🟡 | ~164M pure-Tamil tokens in v1; 38.5% of tokens are `▁` markers (`docs/corpus/DATA_WEIGHTING.md`) | Re-measure on v1+v2 |
| Packing efficiency | 🔴 | per-doc packing drops remainders: a "300M" budget packs **133.6M** train tokens (`docs/training/CONTINUED_PRETRAIN.md` §2) | Fix, then repack |
| Pretraining | 🟡 | Stage 1 val bits/akshara 2.274 vs floor 2.322; Stage 1b (to step 4,881) was due 25 Oct. **No Stage 1b eval is committed.** Long v3 config (5,000 steps, `adhan_slm_nano_cpu_v3_long.yaml`) is committed | Commit `eval_report_nano_stage1b.json` |
| Post-training (SFT / chat) | 🔴 | 0 SFT examples. Response protocol v0 frozen (`docs/behaviour/RESPONSE_PROTOCOL.md`); Sangam pair builder exists (`scripts/build_sangam_sft.py`) | SFT v0, 5–10k examples |
| Safety (Aram) | 🔴 | Harness built (12 principles, 27 prompts); score "not run" (27 Oct) | First critical-only run |
| Export / edge | 🔴 | `scripts/export_onnx.py` and `quantize_model.py` target the removed HF pipeline (`docs/DEV.md` Notes) | Rework for JAX/Orbax |
| Serving | 🟡 | FastAPI with bounded input and 503s when no model is loaded; never validated against a real checkpoint | `adhan serve` on Stage 1b |
| CI | 🟢 | lint / test / type-check workflows; suite 96 passed / 1 skipped in ~90 s (27 Oct) | Consolidate into one `ci.yml` |
| Repo hygiene | 🟡 | 9 open branches awaiting review, 6 of which duplicate work already on `main`; 16 open backlog items | Close the superseded branches |

## 3. Key metrics

| Metric | Now | Target | Source |
|---|---|---|---|
| Token tax vs GPT-4 (cl100k) | **4.21×** | publish, keep Swaram at 1.00× | `docs/tokenizer/TOKEN_TAX_BENCHMARK_V3.md` |
| Fertility (tok/akshara), canonical | ⚠️ 7 different published values (0.304–0.554) | one canonical number per artifact | see archive digest D1 |
| Corpus tokens (all / pure Tamil) | 609.7M / ~164M (v1 only) | 1.5–2B / ≥1B verified | corpus stats, `DATA_WEIGHTING.md` |
| Packed / corpus token ratio | ~0.46 | ≥ 0.95 | `CONTINUED_PRETRAIN.md` |
| bits/akshara (val) | 2.274 (Stage 1) · ~1.955 at step ~2400 (training val) | < 2.322 floor, trending down | `CONTINUED_PRETRAIN.md` §5.3 |
| Sandhi-correct (G4) | 91.5% | ≥ 90% | same |
| Train throughput | ~2,000 tok/s (12-core CPU) · T4 est. 15–25k | ≥ 15k on GPU | `COLAB_T4.md` |
| SFT examples | 0 | 20–50k (nano) | `CAPABILITIES.md` §5 |
| Aram Harness score | not run | > 0.90, 0 CRITICAL | `ARAM_HARNESS.md` |

## 4. In flight

| Change | What | Age | State | Needed to land |
|---|---|---|---|---|
| Six early-Phase-2 branches | Corpus pipeline, tokenizer, nano config, GQA/T4, health/mixture, Sangam ingest | ~2–4 wks | Same work already on `main` | Diff each against `main`; close if empty |
| Tamil normalisation suite | NFC + grapheme checks in CI | 4 days since last update | awaiting review | Review + merge |
| Roadmap sync to the Q4 plan | docs | 13 days | awaiting review | Close as superseded by `factory/` |
| Colab/PyTorch workspace scaffold | notebook + corpus mount | 5 wks | awaiting review | Decide: the JAX T4 notebook already exists (`COLAB_T4.md`) |

## 5. Blockers (oldest first)

| # | Blocker | Since | Blocks | Ask / who |
|---|---|---|---|---|
| B1 | **What Adhan Kutty v1 is** (from-scratch nano vs LoRA/continued pretraining of an open base) is undecided. Part of the Kutty backlog assumes a base model; the roadmap forbids borrowed bases | 6 Oct (backlog filed) | the whole Kutty backlog and the 26 Dec launch | Founder decision by **30 Oct** (`NEXT.md` N2) |
| B2 | Export/quantize does not work with JAX checkpoints | ≥ Aug | phone demo, INT4 build, `yazh-unity`, `illakiya` | Adhan DRI to scope the rework (`NEXT.md` N9) |
| B3 | Training runs on one 12-core CPU box at ~2k tok/s | ongoing | Stage 2/3 timelines | GPU access: the T4 recipe exists; the platform GPU node is the sovereign option |

## 6. Decisions needed

| # | Decision | Options | Recommendation | Decider | By |
|---|---|---|---|---|---|
| D1 | Kutty v1 base | (a) from-scratch adhan-nano + SFT · (b) LoRA on an open base · (c) both, with (a) as the headline | **(a)**. It is the thesis and the differentiator; (b) as an internal baseline only | Founder | 30 Oct |
| D2 | Canonical tokenizer numbers | keep per-doc numbers · one generated table | **One generated table** (`benchmark_token_tax.py` on a pinned sample); docs link to it | Adhan DRI | 29 Oct |
| D3 | Morph-guard in the next tokenizer | keep v3 · v4 with `--morphology-guard` | Decide **after** Stage 1b eval. Changing the tokenizer invalidates all shards and checkpoints | Adhan DRI | 14 Nov |
| D4 | Where to pretrain Stage 2 | CPU box · free T4 · platform GPU node | Platform GPU node if it is up; T4 as fallback | Adhan DRI + infra | 7 Nov |

## 7. Done last iteration (12–27 Oct)

Phase 1 foundation hardening: data-parallel sharding, KV cache, HF revision pinning, round-trip + resume tests · CI back to green (lint, bandit, JAX-less collection) · Aram Harness framework and structured Tamil reasoning protocol · Phase 2 corpus: v2 ingestion (538k new docs), TVA + Wikisource crawlers, per-source failure isolation · Tokenizer v3 + token-tax benchmark + Swaram whitepaper · Capabilities plan + frozen response protocol v0

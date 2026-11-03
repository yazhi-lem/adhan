# CURRENT — adhan · 2026-11-03

> Reset every Monday. Facts only; each number cites its source. The template and rules are in [`README.md`](README.md).

## 1. Headline

**🔴 Red — strategic reset.** The tokenizer is the strongest asset Yazhi has, but the central thesis is still
**unvalidated**: Swaram has never been compared to a Tamil-trained BPE at equal vocab and compute (the whitepaper
admits "no Tamil-trained BPE baseline yet"). The model is a 14.5M-param CPU toy that barely beats a unigram floor;
its Stage 1b generations are not coherent Tamil. The public site still claims "7B-parameter · 22+ Indian languages"
— a ~3-orders-of-magnitude overstatement that must be corrected this week. A weight-sharing regression silently
trained the v3-long run as a 9.2M model until it was caught on 2 Nov. The date with the most risk is **Adhan Kutty v1
on 26 Dec (53 days)**; the two things in its way are the **Kutty base decision** (from scratch vs fine-tune) and the
**absence of a GPU**. Full review: [`docs/reviews/2026-11-03-battle-plan.md`](../docs/reviews/2026-11-03-battle-plan.md).

## 2. Health

| Workstream | RAG | Fact (source) | Next checkpoint |
|---|---|---|---|
| Tokenizer (Swaram v3) | 🔴 | vocab 31,964 · 100% lossless · tax vs GPT-4 4.21×, **but loses to a Tamil-trained SentencePiece BPE: fertility 0.527 vs 0.346 tok/akshara (Swaram uses 1.52× more tokens)** (`docs/reports/ab_swaram_vs_bpe.md`) | Model-level A/B (bits/akshara) or drop the compression claim |
| Tokenizer morphology (G3) | 🔴 | boundary agreement 5.6% (v3 sample) / 24.2% (Stage 1b sample); gate is ≥30%. The morph-guard variant reached 37.5% at +23% tokens (`TOKENIZER_BENCHMARK_V2.md`) | Decide whether v4 uses the guard |
| Corpus scale | 🟡 | ~609.7M swaram tokens (v1 328M + v2 281.7M), 1.87M docs. Target 1.5–2B, so ~35% of the way (Phase 2 totals, 27 Oct) | Raise IndicCorp / mC4 / FineWeb limits |
| Corpus purity | 🟡 | ~164M pure-Tamil tokens in v1; 38.5% of tokens are `▁` markers (`docs/corpus/DATA_WEIGHTING.md`) | Re-measure on v1+v2 |
| Packing efficiency | 🟢 | fixed: `StreamingPacker` carries remainders across documents; the 300M budget now packs **293.57M / 293.57M** tokens (99.9997%) into `nano_300m_s512_v2` (was 133.6M, 0.46×) | Use the v2 shards for Stage 2 |
| Pretraining | 🟢 | Stage 1b (step 4,881) evaluated: val **bits/akshara 1.820** vs floor 2.322 (nll 3.537, ppl 34.36 on the 300m val set) — a clear win over Stage 1's 2.274 (`docs/reports/eval_report_nano_stage1b.json`) | SFT v0 on top of Stage 1b |
| Post-training (SFT / chat) | 🟡 | ~11.2k examples exist: Sangam 9,961 train / 1,303 val / 1,146 test (zero poem overlap) + grounded QA 971 / 124 / 182 (`data/final/sft_*/stats.json`). No SFT run yet; no SFT-vs-base regression gate | SFT v0 with regression gate |
| Safety (Aram) | 🔴 | Harness built (12 principles, 27 prompts); score "not run" (27 Oct) | First critical-only run |
| Export / edge | 🔴 | `scripts/export_onnx.py` and `quantize_model.py` target the removed HF pipeline (`docs/DEV.md` Notes) | Rework for JAX/Orbax |
| Serving | 🟡 | FastAPI with bounded input and 503s when no model is loaded; never validated against a real checkpoint | `adhan serve` on Stage 1b |
| CI | 🟢 | lint / test / type-check workflows; suite **127 passed / 2 skipped** (3 Nov); structure-invariant + remat-grad tests added | Consolidate into one `ci.yml` |
| Repo hygiene | 🔴 | **29 remote branches**, 6 stacked duplicates of work already on `main`; 2 hygiene fixes cherry-picked 3 Nov | Close to <10 branches |
| Thesis validation | 🔴 | A/B run 3 Nov: Swaram beats byte-level BPE (2.49×) but **loses to SentencePiece BPE** (0.346 vs 0.527 tok/akshara). Compression claim not supported; morphological/downstream claim still untested | Model-level A/B (N14b) |

## 3. Key metrics

| Metric | Now | Target | Source |
|---|---|---|---|
| Token tax vs GPT-4 (cl100k) | **4.21×** | publish, keep Swaram at 1.00× | `docs/tokenizer/TOKEN_TAX_BENCHMARK_V3.md` |
| Fertility (tok/akshara), canonical | ⚠️ 7 different published values (0.304–0.554) | one canonical number per artifact | see archive digest D1 |
| Corpus tokens (all / pure Tamil) | 609.7M / ~164M (v1 only) | 1.5–2B / ≥1B verified | corpus stats, `DATA_WEIGHTING.md` |
| Packed / corpus token ratio | **~1.00** (293.57M / 293.57M) | ≥ 0.95 | `data/final/nano_300m_s512_v2/datasheet.json` |
| bits/akshara (val) | **1.820 (Stage 1b, step 4,881)** · 2.274 (Stage 1) · 1.955 at step ~2400 | < 2.322 floor | `docs/reports/eval_report_nano_stage1b.json` |
| Sandhi-correct (G4) | 91.5% | ≥ 90% | same |
| Train throughput | ~2,000 tok/s (12-core CPU) · T4 est. 15–25k | ≥ 15k on GPU | `COLAB_T4.md` |
| SFT examples | ~11.2k (Sangam 9,961 + QA 1,277) | 20–50k (nano) | `data/final/sft_*/stats.json` |
| Aram Harness score | not run | > 0.90, 0 CRITICAL | `ARAM_HARNESS.md` |
| Swaram vs Tamil-BPE (fertility) | **0.527 vs 0.346** (BPE wins) | Swaram wins by ≥10% | `docs/reports/ab_swaram_vs_bpe.json` |
| Swaram vs Tamil-BPE (bits/akshara) | **not measured** | Swaram wins by ≥10% at equal compute | `NEXT.md` N14b |

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
| B1 | **What Adhan Kutty v1 is** (from-scratch nano vs LoRA/continued pretraining of an open base) is undecided. Part of the Kutty backlog assumes a base model; the roadmap forbids borrowed bases | 6 Oct (backlog filed) | the whole Kutty backlog and the 26 Dec launch | Founder decision **overdue**; recommended (c) in `NEXT.md` N2 |
| B2 | Export/quantize does not work with JAX checkpoints | ≥ Aug | phone demo, INT4 build, `yazh-unity`, `illakiya` | Adhan DRI to scope the rework (`NEXT.md` N9) |
| B3 | Training runs on one 12-core CPU box at ~2k tok/s | ongoing | Stage 2/3 timelines | GPU access: the T4 recipe exists; rent a T4/A10 for a weekend (`NEXT.md` N17) |
| B4 | **Thesis not supported on compression**: a Tamil-trained SentencePiece BPE is 1.52× more compressed than Swaram | 3 Nov (A/B) | the entire from-scratch thesis and the tokenizer paper | Run the model-level A/B (N14b); if it also loses, pivot Kutty to a fine-tune |
| B5 | **Public claim wrong**: yazhi.dev says "7B-parameter · 22+ Indian languages" | ≥ Oct | credibility of every public number | Fix this week (`NEXT.md` N5) |

## 6. Decisions needed

| # | Decision | Options | Recommendation | Decider | By |
|---|---|---|---|---|---|
| D1 | Kutty v1 base | (a) from-scratch adhan-nano + SFT · (b) LoRA on an open base · (c) ship (b) as product, keep (a) as research artifact | **(c)**, now strengthened by the A/B: the compression thesis did not survive a Tamil BPE | Founder | **overdue** |
| D2 | Canonical tokenizer numbers | keep per-doc numbers · one generated table | **One generated table** (`benchmark_token_tax.py` on a pinned sample); docs link to it | Adhan DRI | 29 Oct |
| D3 | Morph-guard in the next tokenizer | keep v3 · v4 with `--morphology-guard` | Decide **after** Stage 1b eval. Changing the tokenizer invalidates all shards and checkpoints | Adhan DRI | 14 Nov |
| D4 | Where to pretrain Stage 2 | CPU box · free T4 · platform GPU node | Platform GPU node if it is up; T4 as fallback | Adhan DRI + infra | 7 Nov |
| D5 | Rent a GPU for the A/B and Stage 2 | stay on CPU · rent T4/A10 · platform node | **Rent a T4/A10 for a weekend** (~$10–20); sovereignty is about weights/data, not rented FLOPs | Adhan DRI | 7 Nov |
| D6 | Public claim wording | keep "7B · 22+" · correct to "14.5M research-preview · Tamil" | **Correct it**; link to the model card | Founder + yazhi-dev | 4 Nov |

## 7. Done last iteration (12–27 Oct)

Phase 1 foundation hardening: data-parallel sharding, KV cache, HF revision pinning, round-trip + resume tests · CI back to green (lint, bandit, JAX-less collection) · Aram Harness framework and structured Tamil reasoning protocol · Phase 2 corpus: v2 ingestion (538k new docs), TVA + Wikisource crawlers, per-source failure isolation · Tokenizer v3 + token-tax benchmark + Swaram whitepaper · Capabilities plan + frozen response protocol v0

**2 Nov — critical model regression fixed.** Commit `e6f5a99` hoisted `Block(c)` out of the layer loop, so Flax reused one parameter scope for every layer: the model silently became a weight-shared 9.2M-param stack (not 14.5M) and every older `block_{i}` checkpoint failed to load. Fixed by instantiating `Block(c, name=f"block_{i}")` inside the loop, with a regression test (`test_each_layer_has_its_own_block_params`). The in-flight v3-long run (checkpoints `Block_0` only, 9.2M params) was trained with the bug and is being restarted from scratch.

**2–3 Nov — Stage 1b eval + packing fix + review.** Stage 1b (step 4,881) evaluated: val bits/akshara **1.820** (was 2.274). Cross-document packing fixed (`StreamingPacker`): the 300M budget now packs **99.9997%** of tokens (was 0.46×). Two unmerged hygiene fixes cherry-picked (MLflow loopback bind, `--sangam-root` required). Senior review + battle plan written (`docs/reviews/2026-11-03-battle-plan.md`).

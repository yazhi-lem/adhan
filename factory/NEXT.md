# NEXT — adhan · iteration 2 · 2026-11-03 → 2026-11-17

> **Strategic reset.** Iteration 1 (below) shipped Stage 1b eval and the packing fix, but the
> [3 Nov review](../docs/reviews/2026-11-03-battle-plan.md) reordered the plan around one
> question: **is the from-scratch akshara thesis real?** The P0 set now leads with the
> Swaram-vs-Tamil-BPE A/B (N14) and the correctness harness (N15). The Kutty ADR (N2) is
> overdue and must land this week.

**Iteration goal:** prove or kill the tokenizer thesis with a controlled A/B, make the model
structurally impossible to regress, fix the public claim, and get GPU access — so the 53-day
path to 26 Dec is a real plan, not a hope.

**Capacity:** one full-time DRI ≈ 8 working days per 2-week iteration. The P0 set below is
~7 days; P1 ~6 days. Two people if all P1 is committed; otherwise cut at P1.
**Committed:** P0 = 7 days · P1 = 6 days

| ID | P | Task | DRI | Size | Exit criterion | Depends on | Due |
|---|---|---|---|---|---|---|---|
| N1 | P0 | ✅ **Evaluate Stage 1b.** Run `run_eval` on checkpoint 4,881 and best-val against `nano_300m_s512` val; re-baseline 1,829 on the same val set | Adhan DRI | S (1d) | `docs/reports/eval_report_nano_stage1b.json` committed; bits/akshara **1.820** (was 2.274), G1 0.357 ✅, G3 24.2% ❌, G4 91.5% ✅ recorded in `CURRENT` §3 | Stage 1b finished (verify `done (pretrain).` in log) | 29 Oct |
| N2 | P0 | 📝 **Kutty v1 decision record.** One page: what Kutty v1 is (base, size, tokenizer, SFT scope, target device), what it is not, and the go/no-go bar. Draft recommends option (c) | Founder (decides) · Adhan DRI (drafts) | S (1d) | `docs/adr/0001-kutty-v1.md` merged; the Kutty backlog retitled or closed to match | N1 numbers | **overdue — 4 Nov** |
| N3 | P0 | 🔍 **Publish the tokenizer benchmark**: pinned 355-doc sample manifest, `benchmark_token_tax.py --repro`, a single canonical results table, whitepaper link. Tanglish source scoped: `vishnu-n/Tanglish-Corpus-185k` @ `927f5b4` (CC-BY-4.0); gazetteer-first political sanitization + MuRIL assessment, see `docs/reports/n3_tanglish_source_scope.md`, `docs/reports/muril_base_assessment.md` | Adhan DRI | M (2d) | Script regenerates the V3 table byte-for-byte from a clean checkout, and its default `--out` is that canonical file; README and whitepaper cite that table; post handed to yazhi-dev Media | — | 10 Nov |
| N4 | P0 | **Fix number drift.** Replace the seven fertility values and four morphology values with references to the N3 table; fix the missing `FOUNDER.md` reference in `CAPABILITIES.md` | Adhan DRI | S (0.5d) | `grep -rn "fertility" docs/` shows only values that link to the canonical table | N3 | 10 Nov |
| N5 | P0 | **Correct the public claim on yazhi.dev**: About page says "7B-parameter … 22+ Indian languages" | yazhi-dev DRI (Adhan DRI supplies the wording) | S (0.5d) | yazhi.dev About page matches the model card facts: 14.5M nano, Tamil-first, research stage | — | **4 Nov** |
| N6 | P0 | **Review-queue hygiene.** Diff the branches against `main`, close merged ones (2 hygiene fixes cherry-picked 3 Nov); merge the Tamil normalisation suite; close roadmap-sync and Colab-scaffold | Adhan DRI | S (1d) | < 10 remote branches, ≤ 3 awaiting review, none older than 7 days without one | — | **7 Nov** |
| N7 | P0 | ✅ **Cross-document packing fix.** `StreamingPacker` carries remainders across docs in `freeze_fast_tokenizer.py`; tests prove packed ≥ 95%; repacked the 300M budget | Adhan DRI | M (2.5d) | Test proves packed tokens ≥ 95% (99.9997%); `nano_300m_s512_v2` datasheet committed (293.57M tokens, 561,927 train seqs) | — | 5 Nov |
| N8 | P1 | ✅ **First Aram Harness run** (`--critical-only`) on the best Stage 1b checkpoint, to establish the pipeline and baseline (expected to fail pre-SFT) | Adhan DRI | S (1d) | `docs/reports/aram_report_stage1b.json` committed; **score 0.13, 7 CRITICAL** recorded in `CURRENT` §3 | N1 | 4 Nov |
| N9 | P1 | **Scope the export rework**: Orbax → SafeTensors/HF, ONNX, INT8. Spike the INT8 path for nano and measure the size and quality delta | Adhan DRI | M (3d) | Design note + a working INT8 export of the Stage 1b checkpoint with ppl delta measured | N2 | 7 Nov |
| N10 | P1 | ✅ **SFT v0 dataset (Sangam half).** `build_sangam_sft.py` → tiṇai/kalam/urai/citation pairs, split by poem, response protocol v0 | Adhan DRI | M (2d) | `data/final/sft_v0_sangam` (**9,961 train / 1,303 val / 1,146 test**) + `docs/reports/datasheet_sft_v0_sangam.json`; ≥ 5k examples; **zero poem overlap** (`stats.json`) | — | 7 Nov |
| N11 | P1 | ✅ **Validate serving on a real checkpoint**: `adhan serve` with Stage 1b; measure `/generate` p50/p95 for 48 tokens on CPU | Adhan DRI | S (1d) | **p50 1,757 ms / p95 2,540 ms** on CPU in `CURRENT` §3; `docs/reports/serving_stage1b.json` | N1 | 6 Nov |
| N12 | P1 | **Kutty target device**: pick the ₹10–15k reference phone and runtime (ONNX Runtime Mobile / llama.cpp-class) | Adhan DRI | S (1d) | Device + runtime recorded in the N2 ADR; procurement started | N2 | 7 Nov |
| N13 | P0 | **Stage 2 / A/B compute: book a GPU.** Rent a T4/A10 for a weekend (~$10–20) or use the platform node; run the existing T4 config and record tok/s | Adhan DRI + infra | S (0.5d) | Host named, window booked, cost noted, T4 config run once (`CURRENT` D4/D5) | — | 7 Nov |
| N14a | P0 | ✅ **Tokenizer-level A/B.** Swaram vs byte-level and SentencePiece BPE, same 300M-char corpus, same 31,964 vocab, 2,000 held-out docs | Adhan DRI | S (1d) | **Swaram loses to the strong BPE: 0.527 vs 0.346 tok/akshara** (1.52× more tokens). `docs/reports/ab_swaram_vs_bpe.{json,md}` committed | — | 3 Nov |
| N14b | P0 | **Model-level A/B (the decisive test).** Equal-compute nano training with Swaram vs the SentencePiece BPE; compare held-out bits/akshara + 20-prompt coherence. Compression ≠ quality | Adhan DRI | M (3d) | `docs/reports/ab_model_swaram_vs_bpe.json`; if Swaram does not win, D1 pivots to a fine-tune | N13, N14a | 14 Nov |
| N15 | P0 | ✅ **Structure-invariant model test in CI.** Fuzzed config matrix asserts `n_layers` distinct blocks, param count == `approx_params()`, orbax init→save→load round-trip, and finite remat gradients | Adhan DRI | S (1d) | `gqa_tests.py` (19 tests); a `Block` hoist or shared scope fails CI. Also fixed a latent remat tracer leak (`nn.remat` wraps the class, not the instance) | — | 7 Nov |
| N16 | P1 | **50-prompt human coherence eval + SFT-vs-base regression gate.** Rubric scored 1–5, published next to the no-SFT baseline | Adhan DRI | M (2d) | `docs/reports/coherence_stage1b.json`; gate wired into the SFT runner | N1 | 14 Nov |
| N17 | P1 | **Reconcile corpus numbers + fix the `▁`-marker tax.** Measure pure-Tamil tokens, post-dedup docs, and actual tokens in the v2 shards; quantify the 38.5% marker overhead | Adhan DRI | S (1d) | One measured corpus table in `CURRENT` §3; no two docs disagree | — | 14 Nov |

## Stretch (only if committed work is done)

CI consolidation into one `ci.yml` · LSH banding for near-dups · INT8 export spike (N9).

## Explicitly not this iteration

- Growing past nano, adding special tokens, or a tokenizer change (`CAPABILITIES.md` §7, "do not do yet") **until N14 reports**.
- Multilingual Layer A work (Kannada/Telugu).
- Distilling from a teacher model before the licence terms for its outputs are written down (`RUNWAY` R5).
- Building `tiny`/`mini` before nano is coherent and ≥1B verified pure-Tamil tokens exist.

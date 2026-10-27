# NEXT — adhan · iteration 1 · 2026-10-28 → 2026-11-10

**Iteration goal:** ship the public tokenizer evidence, turn Stage 1b into a measured baseline, and settle what
Kutty v1 is, so that the 60-day path to 26 Dec is concrete and staffed.

**Capacity:** _set this to measured velocity × 2 weeks_. As a rule of thumb, one full-time DRI covers ~8 working days;
the P0+P1 set below is ~17 days, so it needs two people or a cut at P1.
**Committed:** P0 = 8.5 days · P1 = 8.5 days

| ID | P | Task | DRI | Size | Exit criterion | Depends on | Due |
|---|---|---|---|---|---|---|---|
| N1 | P0 | **Evaluate Stage 1b.** Run `run_eval` on checkpoint 4,881 and best-val against `nano_300m_s512` val; re-baseline 1,829 on the same val set | Adhan DRI | S (1d) | `docs/reports/eval_report_nano_stage1b.json` committed; bits/akshara, G1–G4 and the 1,829 delta recorded in `CURRENT` §3 | Stage 1b finished (verify `done (pretrain).` in log) | 29 Oct |
| N2 | P0 | **Kutty v1 decision record.** One page: what Kutty v1 is (base, size, tokenizer, SFT scope, target device), what it is not, and the go/no-go bar | Founder (decides) · Adhan DRI (drafts) | S (1d) | `docs/adr/0001-kutty-v1.md` merged; the Kutty backlog retitled or closed to match | N1 numbers | 30 Oct |
| N3 | P0 | **Publish the tokenizer benchmark**: pinned 355-doc sample manifest, `benchmark_token_tax.py --repro`, a single canonical results table, whitepaper link | Adhan DRI | M (2d) | Script regenerates the V3 table byte-for-byte from a clean checkout, and its default `--out` is that canonical file; README and whitepaper cite that table; post handed to yazhi-dev Media | — | 30 Oct (campaign 31 Oct) |
| N4 | P0 | **Fix number drift.** Replace the seven fertility values and four morphology values with references to the N3 table; fix the missing `FOUNDER.md` reference in `CAPABILITIES.md` | Adhan DRI | S (0.5d) | `grep -rn "fertility" docs/` shows only values that link to the canonical table | N3 | 31 Oct |
| N5 | P0 | **Correct the public claim on yazhi.dev**: About page says "7B-parameter … 22+ Indian languages" | yazhi-dev DRI (Adhan DRI supplies the wording) | S (0.5d) | yazhi.dev About page matches the model card facts: 14.5M nano, Tamil-first, research stage | — | 30 Oct |
| N6 | P0 | **Review-queue hygiene.** Diff the six early-Phase-2 branches against `main` and close the ones already merged; review and merge the Tamil normalisation suite; close the roadmap-sync and Colab-scaffold branches or land them | Adhan DRI | S (1d) | ≤ 3 branches awaiting review, none older than 7 days without one | — | 31 Oct |
| N7 | P0 | **Cross-document packing fix.** Carry remainders across doc boundaries (with EOS separators) in `freeze_fast_tokenizer.py` / `prepare_slm_corpus.py`; add a test; repack the 300M budget | Adhan DRI | M (2.5d) | Test proves packed tokens ≥ 95% of corpus tokens; new `nano_300m_s512_v2` datasheet committed | — | 5 Nov |
| N8 | P1 | **First Aram Harness run** (`--critical-only`) on the best Stage 1b checkpoint, to establish the pipeline and baseline (expected to fail pre-SFT) | Adhan DRI | S (1d) | `docs/reports/aram_report_stage1b.json` committed; score in `CURRENT` §3 (replaces "not run") | N1 | 4 Nov |
| N9 | P1 | **Scope the export rework**: Orbax → SafeTensors/HF, ONNX, INT8. Spike the INT8 path for nano and measure the size and quality delta | Adhan DRI | M (3d) | Design note + a working INT8 export of the Stage 1b checkpoint with ppl delta measured | N2 | 7 Nov |
| N10 | P1 | **SFT v0 dataset (Sangam half).** `build_sangam_sft.py` → tiṇai/kalam/urai/citation pairs, split by poem (15% held out), response protocol v0 | Adhan DRI | M (2d) | `data/final/sft_v0_sangam` + datasheet; ≥ 5k examples; zero poem overlap between train and eval (tested) | — | 7 Nov |
| N11 | P1 | **Validate serving on a real checkpoint**: `adhan serve` with Stage 1b; measure `/generate` p50/p95 for 48 tokens on CPU | Adhan DRI | S (1d) | Numbers in `CURRENT` §3 | N1 | 6 Nov |
| N12 | P1 | **Kutty target device**: pick the ₹10–15k reference phone and runtime (ONNX Runtime Mobile / llama.cpp-class) | Adhan DRI | S (1d) | Device + runtime recorded in the N2 ADR; procurement started | N2 | 7 Nov |
| N13 | P1 | **Stage 2 compute decision** (`CURRENT` D4) and booking | Adhan DRI + infra | S (0.5d) | Host named, window booked, cost/day noted | — | 7 Nov |

## Stretch (only if committed work is done)

CI consolidation into one `ci.yml` · LSH banding for near-dups.

## Explicitly not this iteration

- Growing past nano, adding special tokens, or a tokenizer change (`CAPABILITIES.md` §7, "do not do yet").
- Multilingual Layer A work (Kannada/Telugu).
- Distilling from a teacher model before the licence terms for its outputs are written down (`RUNWAY` R5).

# ADR 0001 — Adhan Kutty v1: base, tokenizer, scope, and ship bar

**Status:** Proposed — awaiting Founder sign-off
**Date:** 2026-11-03
**Decider:** Founder · **Drafters:** Adhan DRI
**Supersedes:** the undecided "from scratch vs fine-tune" question open since 6 Oct
**Related:** `factory/NEXT.md` N2 · `factory/RUNWAY.md` §6 · `docs/reviews/2026-11-03-battle-plan.md` §3

---

## 1. Context

Kutty v1 is due **26 Dec 2026 (53 days)**. The facts that constrain the decision:

| Fact | Value | Source |
|---|---|---|
| Best from-scratch checkpoint | `adhan-nano`, 14.5M params, Stage 1b step 4,881, val **1.820 bits/akshara** | `docs/reports/eval_report_nano_stage1b.json` |
| Stage 1b generation quality | short Tamil is not coherent; no SFT run yet | `factory/CURRENT.md` §1 |
| Tokenizer A/B | Swaram **loses** to a Tamil-trained SentencePiece BPE: 0.527 vs 0.346 tok/akshara (1.52× more tokens) | `docs/reports/ab_swaram_vs_bpe.md` |
| Morphology gate G3 | boundary agreement 5.6–24.2%, gate ≥ 30% | `docs/tokenizer/TOKENIZER_BENCHMARK_V2.md` |
| GPU | none; CPU host sustains ~2,000 tok/s | `factory/CURRENT.md` §3, B3 |
| SFT data | ~11.2k examples (Sangam 9,961 + grounded QA 1,271), never trained | `data/final/sft_*/stats.json` |
| Export | ONNX/quantize still target the removed HF pipeline | `factory/CURRENT.md` B2 |
| Public claim | yazhi.dev says "7B-parameter · 22+ Indian languages" — false | `factory/CURRENT.md` B5 |

The from-scratch thesis still has one untested claim: that *akshara-native tokens produce a
better Tamil model at equal compute*, independent of raw compression. That is `N14b`.
It needs a GPU and is due 14 Nov. Kutty v1 cannot wait for a positive result to start.

## 2. Decision (recommended)

Adopt **option (c): ship a fine-tune-based product, keep the from-scratch model as the
research artifact.**

1. **Kutty v1 product model** is a LoRA / continued-pretraining adaptation of an open,
   permissively licensed **0.5–2B Tamil-capable base**, using that base's own tokenizer.
   Final base selection is gated on a redistribution-compatible licence and a Tamil
   tokenizer/context check before any training spend.
2. **Kutty v1 is not** the from-scratch `adhan-nano`. `adhan-nano` + Swaram remain the
   public research contribution (tokenizer paper, ablations, checkpoints).
3. **SFT scope for v1:** Sangam and grounded QA first, with citations and refusal. Add
   conversation only if the Sangam + QA sets clear their gates in time. Assistant-turn loss
   only; 2–3 epochs; a no-SFT regression gate is mandatory.
4. **Target device:** a ₹10–15k Android reference phone, INT8, via ONNX Runtime Mobile or a
   llama.cpp-class runtime. The exact device and runtime are recorded by N12 before
   procurement.
5. **Tokenizer for Kutty v1:** the chosen base's tokenizer. Swaram is used in Kutty v1
   **only if** N14b shows ≥ 10% lower held-out bits/akshara at equal compute. A tokenizer
   swap after training starts invalidates the run.
6. **Go/no-go bar:** the ship bars in `factory/RUNWAY.md` §6. If they are missed, use the
   cut list — last resort is a **research preview** (tokenizer + base checkpoint + model
   card), not a lower quality gate.

## 3. Why (c), not (a) or (b)

- **Schedule:** from-scratch nano has never passed an SFT or coherence gate. A 0.5–2B
  fine-tune starts from a model that already produces grammatical Tamil, so the remaining
  work is data, alignment, export, and evaluation — the work a 26 Dec launch actually needs.
- **Evidence:** the compression argument for Swaram did not survive its first controlled
  test. Option (a) now carries both schedule risk and an unproven thesis.
- **Sovereignty:** sovereignty is about owning the weights, data, and evaluation, not about
  refusing every pretrained base. A permissively licensed base plus owned SFT data,
  tokenizer research, and on-device runtime is a defensible sovereign artifact.
- **Optionality:** (c) does not kill (a). If N14b supports Swaram, the research line
  continues and can become the Kutty v2 foundation.

### MuRIL is not a candidate base

`google/muril-base-cased` was assessed (2026-11-03). It is a 237.6M-parameter
BERT-base **encoder** (Apache-2.0, 17 Indian languages, 512-token context). It has no
autoregressive decoder, so it cannot answer prompts, write Tamil, or absorb the Kutty
SFT/chat tasks. It does not change D1. Its useful role is auxiliary: a fine-tuned
political/abuse classifier for the Tanglish eval source, a retrieval encoder, and an
encoder-only XNLI baseline. See `docs/reports/muril_base_assessment.md`.

## 4. What this is not

- Not permission to use a base whose licence forbids redistribution or commercial use.
- Not permission to drop the Tamil evaluation suites or the Aram harness.
- Not a claim that Kutty v1 is "from scratch" or "7B". Public copy must say
  **research preview, Tamil-first, fine-tuned open base** (exact wording owned by N5).
- Not a tokenizer decision for v2/v3 — Swaram v4 and special tokens remain gated on N14b
  and D3.

## 5. Consequences

**Positive:** a realistic 26 Dec path; SFT and export work can start against a model that
already speaks Tamil; the research thesis gets a fair, separate test.

**Negative / accepted risk:** the product no longer demonstrates the from-scratch thesis;
base licence and base quality become new gates; the team carries two narratives (product
vs research) and must keep them clearly separated in public copy.

## 6. Follow-ups required to execute

| # | Follow-up | Owner | Gate |
|---|---|---|---|
| F1 | Founder signs this ADR or records a different choice | Founder | before training spend |
| F2 | Select the base: licence, Tamil tokenizer, context, size, INT8 support | Adhan DRI | before N14b/Stage 2 |
| F3 | Run N14b (Swaram vs BPE at model level) | Adhan DRI | 14 Nov |
| F4 | Rework export for the chosen base/runtime (N9) | Adhan DRI | 7 Nov spike |
| F5 | Pick reference phone + runtime (N12) | Adhan DRI | 7 Nov |
| F6 | Correct the public claim on yazhi.dev (N5) | Founder + yazhi-dev | 4 Nov |

If F1 rejects option (c), this ADR is superseded and option (a) must be re-costed against
the 26 Dec date with a named GPU and a named SFT + coherence gate.

# Swaram vs a Tamil-trained BPE — A/B result (N14a)

**Date:** 2026-11-03 · **Script:** `scripts/ab_swaram_vs_bpe.py` · **Raw result:** [`ab_swaram_vs_bpe.json`](ab_swaram_vs_bpe.json)

## Setup

| | |
|---|---|
| Corpus | `data/raw/expanded` (1.34M docs, 1.06B chars) |
| BPE training sample | 300,000,145 chars (358,117 docs) |
| Held-out | 2,000 docs (deterministic hash split, same rule as the shards) |
| Vocab | 31,964 (identical for all three tokenizers) |
| Akshara count | tokenizer-independent (`indic_akshara.segment`, Tamil) |

Two baselines were trained on the **same Tamil corpus**:

- **byte** — GPT-2-style byte-level BPE. The direct control for "the same tokenizer
  architecture as cl100k/o200k, retrained on Tamil".
- **metaspace** — SentencePiece-style BPE over Unicode with a `▁` word mark
  (IndicBERT/MuRIL-style). The **strong** baseline a reviewer will ask for.

## Result

| tokenizer | fertility (tok/akshara) | tokens/word | chars/token | tax vs Swaram |
|---|---|---|---|---|
| **Swaram v2** | **0.5273** | 2.593 | 3.465 | 1.00× |
| BPE byte 32k | 1.3145 | 6.464 | 1.390 | 2.49× |
| **BPE metaspace 32k** | **0.3458** | 1.701 | 5.283 | **0.66×** |

**Overall verdict (keyed off the strongest baseline): `thesis_not_supported_on_this_sample`.**
Swaram beats the byte-level BPE (2.49× tax) but **loses to the SentencePiece BPE**:
the strong baseline packs the same text into **0.66×** the tokens, i.e. Swaram uses
~1.52× more tokens than a properly-trained Tamil BPE.

## What this means

1. **The "token tax" framing is dead as stated.** The white paper's 4.21× is a tax
   against an *English-fitted* tokenizer, not a property of Tamil. A tokenizer
   trained on Tamil at the same vocab is more compressed than Swaram.
2. **Compression is not the thesis.** The remaining case for Swaram is
   *morphological* alignment (agglutination-aware tokens) and downstream sample
   efficiency — not raw token counts. That case currently fails its own gate
   (G3, morphology agreement 5.6–24.2%, target ≥ 30%).
3. **The model-level A/B is now the decisive test.** A more compressed tokenizer
   is not automatically a better model: each token may be harder to predict.
   The next step (`N14b`) is equal-compute nano training with each tokenizer and
   a comparison of **bits/akshara** on held-out data, plus a 20-prompt coherence
   read.
4. **This strengthens the Kutty pivot.** If the from-scratch akshara thesis does
   not win at the model level either, Kutty v1 should be a fine-tune of an open
   base (battle plan D1 option c), with Swaram kept as a research/tokenizer
   contribution rather than the product's foundation.

## Reproduce

```bash
python scripts/ab_swaram_vs_bpe.py \
    --corpus data/raw/expanded \
    --swaram-tokenizer data/final/tamil_slm_v2 \
    --train-chars 300000000 --val-docs 2000 \
    --out docs/reports/ab_swaram_vs_bpe.json
```

Deterministic given the corpus snapshot and `--seed 0`.

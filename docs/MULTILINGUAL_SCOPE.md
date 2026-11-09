# Future scope: competing with frontier models as a multilingual (Indic) SLM

An honest analysis written against the numbers this repo can actually produce today
(§1), not against aspirations. The short version: **we cannot out-scale frontier
models, and we should not try. We can out-tokenize them on Indic text, and that is a
real, measurable, defensible advantage — if we build the multilingual path on the
akshara layer instead of on vocabulary growth.**

---

## 1. Where we actually stand (measured)

From `scripts/benchmark_tokenizer_v2.py` on 200 held-out archive documents:

| Artifact | Vocab | Fertility (tok/akshara) | Lossless | `<unk>` docs | Encode | Morph. agreement |
|---|---|---|---|---|---|---|
| swaram v1 (python, 12k, no bytes) | 12,000 | 0.381 | 98.0% | 4/200 | 13 kchar/s | 22.7% |
| swaram v2 (rust, 32k, byte fallback) | 31,964 | 0.304 | 100% | 0 | 2,646 kchar/s | 13.6% |
| swaram v2 + morphology guard | 31,964 | 0.374 | 100% | 0 | 2,707 kchar/s | **37.5%** |

Corpus: 1,334,035 documents / 1.06G characters / **~293–328M swaram tokens**, zero
exact duplicates. Packed artifacts: full archive (293.5M tokens) and a 30M-token
shard set for fast iteration.

Two facts drive everything below:

1. **Fertility < 1 token per akshara.** A byte-level BPE spends ~1.39 tokens per
   akshara on Tamil (`docs/SWARAM_BENCHMARK.md`). At 0.30–0.37 we encode the same
   text in **roughly a quarter to a third of the tokens**, so every context window,
   every KV-cache byte and every training step buys 3–4× more Tamil.
2. **Morphology is a dial, not a constant.** The guard moved boundary agreement
   13.6% → 37.5% at +23% tokens. Frontier tokenizers never expose this tradeoff;
   ours makes it a CLI flag (`--morphology-guard`).

What we do *not* have: a frontier-scale checkpoint, RLHF, tool use, long context, or
benchmark coverage beyond tokenizer + tiny CPU runs.

---

## 2. Why the akshara layer is the multilingual lever

Every Indic language we care about is written in a **Brahmic abugida**: consonant +
inherent vowel, matras for the rest, virama to suppress. `indic_akshara.py` implements
that rule once for Tamil, Devanagari, Bengali, Gurmukhi, Gujarati, Odia, Telugu,
Kannada, Malayalam and Sinhala — adding a script is a table row, not a tokenizer.

| | Tamil (Dravidian) | Devanagari (Indo-Aryan) |
|---|---|---|
| Base akshara inventory | 404 (curated, incl. Grantha + āytam) | 582 (derived: 45 C × 13 matra forms) |
| Conjuncts | ✗ (virama splits) | ✓ (क्ष, त्र, ज्ञ — halant joins) |
| Sandhi character | agglutinative suffixes, glide insertion | external sandhi + postpositions |
| Byte fallback need | Grantha, āytam, digits | nukta forms, Vedic marks |

Consequences for a multilingual prototype:

- **Vocabulary cost is linear in aksharas, not in subwords.** A new Dravidian
  language is ≈0.6–1.0k base aksharas (plus script-specific extras), versus 8–16k
  subword merges for BPE. Four Dravidian languages fit in the budget one BPE
  language spends rediscovering morphology.
- **Akshara indices are structurally isomorphic across scripts.** The 4th akshara of
  a row is "ka + i" in Tamil, Telugu, Kannada and Malayalam with different codepoints.
  That allows a **shared akshara-slot embedding** (one row per phonological akshara,
  per-script surface form), which is a genuine cross-lingual transfer mechanism that
  byte-level BPE cannot express — BPE spaces are language-idiosyncratic by construction.
- **Byte fallback makes the multilingual story cheap and safe.** Any script, any
  symbol, any emoji round-trips losslessly (verified across 10 scripts + Latin +
  digits + emoji). A Tamil-first model is never *blocked* from foreign text; it just
  pays byte prices for it, which is the correct incentive.

---

## 3. Where a frontier model still wins (be specific, or the plan is fiction)

| Axis | Frontier (Gemma/Llama/Sarvam-class) | Adhan today | Realistic verdict |
|---|---|---|---|
| Pretraining tokens | 10^12–10^13, multilingual | 2.9×10^8 Tamil | **Never** close this gap with one machine |
| Reasoning / instruction following | RLHF/DPO + massive SFT | none | Not competitive for general chat |
| Long context | 128k–1M | 1,024 | Not competitive |
| Indic tokenizer efficiency | ~1.2–1.5 tok/akshara (BPE/SP) | **0.30–0.37** | **We win by 3–4×** |
| Lossless text handling | usually lossy normalization | byte-exact | We win |
| Morphology awareness | implicit, unmeasurable | explicit, tunable, tested | We win (measurable) |
| On-device footprint | 1–8B params | 8M params (nano) | We win where size matters |
| Data sovereignty / licensing | vendor terms | own corpus + Apache/MIT tooling | We win for government/education |

The honest strategic read: **a 10^8-token Tamil model will not beat a 10^12-token
frontier model at general reasoning, ever.** It can beat it at *cost per Tamil token*,
*offline deployability*, and *domain fidelity* (classical Tamil, school curriculum,
government forms) where frontier models are weakest and least interested.

---

## 4. The one metric to publish

Perplexity is not comparable across tokenizers. **Bits per akshara** is:

```
bits/akshara = (nll_per_token / ln 2) × (tokens / akshara)
```

Because our fertility is ~0.3 and a BPE model's is ~1.4, a model with *worse*
per-token perplexity can still have better bits-per-akshara. That is the number that
makes the tokenizer advantage visible instead of rhetorical, and it is why
`EVAL_TAMIL.md` already insists on per-akshara normalisation when comparing to
distilgpt2. Publish it alongside: fertility, lossless rate, morphology agreement,
and MILU/IndicQA/XNLI accuracy from `bench/tamil-bench`.

---

## 5. Staged plan

**Phase A — Tamil-only baseline (now).**
Pretrain `adhan-nano` (8M) on the 30M-token shard set, then the full 293M. Report
bits/akshara against the classical unigram floor (`run_eval`), plus sandhi and
boundary-agreement probes. Gate: beats the floor comfortably, morphology guard keeps
agreement ≥35%.

**Phase B — Tamil + English code-switch.**
Real Tamil text is code-mixed ("இது ஒரு AI model தான்"). Byte fallback already makes
this lossless; the work is data (Tanglish corpora, `transliterate.azhagi` in the
open-tamil bridge) and measuring the byte-share of each document as a quality signal.

**Phase C — Dravidian cluster (ta / te / kn / ml).**
Shared akshara-slot embeddings, per-language merge layers. This is the step where the
architecture, not scale, is the differentiator: four scripts in one vocab for less
than one BPE language. Gate: cross-lingual transfer (train ta → eval te/kn/ml
bits-per-akshara) beats monolingual-only baselines.

**Phase D — Indo-Aryan (hi / bn) via Aksharam.**
Same Layer A with `join_conjuncts=True`. Expect *higher* fertility than Tamil because
Devanagari conjuncts multiply the akshara inventory and Hindi's morphology is less
agglutinative; that is a measurable hypothesis, not a hope.

**Phase E — Evaluation honesty.**
`bench/tamil-bench` (MILU-Tamil, IndicQA, XNLI) pointed at our OpenAI-compatible
server, with the frontier numbers from the same harness as the comparison set. Report
the loss on general benchmarks, not just the wins on tokenizer metrics.

---

## 6. Risks

- **Tokenizer advantage ≠ model quality.** Compression buys context and cost, not
  reasoning. Over-indexing on fertility is the most likely way to fool ourselves.
- **Suffix tables are a proxy.** `morphology.SUFFIXES` covers productive endings, not
  a full analyser; agreement numbers are a signal, not ground truth. The open-tamil
  stemmer/sandhi checker in `eval/` remains the external oracle.
- **Aryan conjuncts.** Devanagari's virama chaining (क्ष) means akshara-atomicity can
  produce long, sparse units. Fertility may approach or exceed 1.0 on Hindi.
- **Scale ceiling.** 293M tokens is a legitimate small-model corpus and a rounding
  error for a frontier model. Scope claims accordingly: Tamil-first SLM, edge and
  domain use, not a general assistant.

---

## 7. Bottom line

The defensible claim is narrow and strong:

> An akshara-native tokenizer encodes Tamil in ~0.30–0.37 tokens per akshara with
> byte-exact fidelity and tunable morphology, and extends to nine more Indic scripts
> by table entry — which makes a Tamil-first SLM 3–4× cheaper per token than a
> byte-level BPE frontier stack, with a shared cross-script embedding space that BPE
> cannot represent.

Everything beyond that — beating frontier models at general reasoning — is a data and
compute question we are not currently equipped to win, and should not pretend
otherwise in public claims.

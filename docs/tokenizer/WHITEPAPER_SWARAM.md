# Swaram: A Script-Native Tokenizer for Tamil

**Removing the Tamil token tax in language models**

*Yazhi / Adhan SLM — Technical Whitepaper v1.0*

---

## Abstract

Subword tokenizers designed for English impose a **token tax** on Tamil: the
same text is represented with several times more tokens than a script-native
tokenizer needs, inflating training compute, inference cost, latency, and
shrinking the effective context window. We present **Swaram**, a two-layer
tokenizer built on the Tamil *akshara* (grapheme cluster) rather than on
English-motivated byte pairs.

Swaram combines (A) a deterministic, lossless akshara segmentation over a closed
inventory of Tamil characters, and (B) a learned BPE merge layer over aksharas,
with optional UTF-8 byte fallback for out-of-inventory characters. On a
deterministic sample of a 538k-document Tamil corpus we measure:

* **fertility 0.506 tokens/akshara** (target < 1.15), **100% round-trip**;
* **1.00× token tax**, against **4.21×** for GPT-4 (`cl100k_base`),
  **1.21×** for GPT-4o (`o200k_base`), **3.26×** for code points, and
  **8.56×** for raw UTF-8 bytes;
* **178 Tamil words** fit in a 512-token context, versus **42** under GPT-4;
* **~2.5 Mchar/s** encoding throughput on a single CPU core.

The tax is paid three times over — in pretraining compute, in API cost, and in
the amount of Tamil a model can actually see. Swaram removes it by making the
akshara the atomic unit of representation.

---

## 1. Introduction

Tamil is an agglutinative, diglossic Dravidian language written in a
syllabic-alphabetic script. A Tamil word can carry several morphemes, and sandhi
(*puṉarcci*) fuses adjacent words at their boundaries. The script is organised
around the **akshara**: a consonant (or vowel) plus its optional vowel sign
(*matra*) and/or *pulli* (virama). An akshara is close to a syllable and is the
natural unit of Tamil orthography.

Modern language models tokenize with Byte-Pair Encoding (BPE) trained
predominantly on English. Tamil characters are then split into byte fragments or
partial aksharas, producing a large number of tokens per word. Three costs
follow:

1. **Compute.** A fixed Tamil corpus requires *tax*× more training tokens.
2. **Money and latency.** Every generated Tamil token is billed and decoded.
3. **Context.** A window of fixed size holds *1/tax* as much Tamil.

We call this multiplier the **token tax**:

```
tax(tokenizer T) = tokens_T(text) / tokens_swaram(text)
```

Swaram is designed so that `tax = 1.00` by construction: representation is
akshara-first, and no English byte-pair statistics constrain Tamil.

### Contributions

* A two-layer akshara-first tokenizer with a closed, lossless inventory.
* An empirical characterisation of the Tamil token tax against widely deployed
  tokenizers (GPT-4, GPT-4o, code-point, UTF-8 byte).
* A reproducible artifact and benchmark harness shipped with the Adhan SLM.

---

## 2. Background

### 2.1 Tamil orthography

The Tamil block is U+0B80–U+0BFF. An akshara consists of a base character
(a vowel, a consonant, or the *aytham* ஃ) followed by zero or more combining
marks: the vowel signs U+0BBE–U+0BCC and the virama/pulli U+0BCD. Tamil also
uses Grantha consonants (ஜ ஷ ஸ ஹ) and ligatures (க்ஷ, ஸ்ரீ) for loanwords.

This structure is *regular*: segmentation does not require a learned model and
is lossless. That is the key property English-centric tokenizers do not exploit.

### 2.2 Agglutination and sandhi

Tamil morphology is suffix-rich (`-கள்` plural, `-இல்` locative, `-உடைய`
genitive, `-கொண்டு` instrumental, `-இருந்தேன்` past progressive). Sandhi alters
the boundary between words (`கடல் + கரை → கடற்கரை`). A tokenizer that splits
inside an akshara destroys the information a morphology-aware model needs; a
tokenizer that merges whole morphemes can help the model generalise.

### 2.3 The token tax

For an English-centric tokenizer, Tamil text is represented as fragments of
UTF-8 bytes. A 3-byte akshara may become 1–6 tokens depending on the BPE merge
table. Because the table was learned on English byte statistics, Tamil sequences
are rarely merged, and the tax is large. Code-point tokenization (one token per
Unicode scalar) is better but still pays ~3× because a Tamil akshara is often
2–3 code points, and it ignores morphology entirely.

---

## 3. Swaram: design

Swaram has two lossless layers and a fallback.

### 3.1 Layer A — akshara segmentation

Input text is segmented deterministically into aksharas:

* a base character starts a new akshara;
* any following combining marks (matras, pulli) attach to it;
* the join is exactly lossless: `"".join(segment(text)) == text`.

The closed inventory has **404** entries: 12 vowels, the aytham, 18 pure
consonants with pulli, the consonant+vowel ligatures, and Grantha characters.
Anything outside the inventory is handled by Layer C (below), never dropped.

Because segmentation is deterministic, it is O(n) and needs no model. In the
current release the fast path is implemented in Rust (`tokenizer.json` artifact);
a pure-Python reference implementation is kept for testing and parity checks.

### 3.2 Layer B — morpheme merges over aksharas

A BPE layer is trained **over aksharas, not bytes**. The vocabulary starts from
the closed inventory plus special tokens and a word-boundary marker, and merges
the most frequent adjacent akshara sequences. This lets the tokenizer represent
common Tamil morphemes and suffixes as single tokens (`-கள்`, `-இல்`, `-உடைய`)
while never cutting through an akshara.

The v3 artifact uses a **31,964-token vocabulary** and **28,549 merges**,
learned from a 300M-character stratified sample of the corpus.

### 3.3 Layer C — byte fallback

Rare characters (digits, punctuation, foreign scripts, emoji, historical
characters) fall back to UTF-8 byte tokens `<0xNN>`. This makes the tokenizer
**total**: every string round-trips, and `<unk>` never appears for arbitrary
input. In our evaluation, 100% of documents round-trip byte-exactly.

### 3.4 Optional morphology guard

Layer B can be constrained to reject merges that cut across known Tamil
stem/suffix boundaries (`--morphology-guard`). This trades some compression for
linguistically aligned boundaries. It is implemented and measured but not yet
enabled in the shipped v3 artifact (see §7).

---

## 4. Method

### 4.1 Corpus

The v3 tokenizer is trained on the Adhan Tamil corpus `data/raw/expanded_v2`:
**538,562 documents, 901.6M characters**, drawn from mC4, IndicCorp v2,
FineWeb-2, Tamil Wikipedia, classical Tamil texts, and digitised libraries
(Tamil Virtual Academy, Tamil Wikisource). Documents are deduplicated
exactly and near-duplicate (banded LSH) and quality-gated on Tamil-character
ratio, length, repetition and boilerplate.

### 4.2 Tokenizer training

* merge sample: 300M characters, stratified by document (stride sampling);
* vocabulary target: 32,000 (realised: 31,964);
* minimum merge frequency: 2;
* byte fallback: enabled;
* sequence length for the packed shards: 512.

Training took **172 s**; packing **258.3M tokens** into 338,758 train
sequences and 6,878 validation sequences took **129 s**.

### 4.3 Evaluation protocol

Two complementary benchmarks, both shipped as scripts:

* **`benchmark_token_tax.py`** — same held-out documents, compares Swaram
  against GPT-4 (`cl100k_base`), GPT-4o (`o200k_base`), code points and UTF-8
  bytes. Reports fertility, tokens/word, chars/token, tax, words fitting in a
  512-token context, round-trip, throughput.
* **`benchmark_tokenizer_v2.py`** — compares frozen Swaram artifacts on
  fertility, lossless rate, `<unk>` documents, throughput and morphological
  boundary agreement.

The akshara denominator is produced by the Swaram segmenter, so the *tax*
comparison is like-for-like across tokenizers.

---

## 5. Results

### 5.1 Token tax (355 held-out documents)

| tokenizer | fertility (tok/akshara) | tokens/word | tax vs Swaram | Tamil words in 512 ctx | round-trip |
|---|---|---|---|---|---|
| **Swaram v3** | **0.554** | **2.874** | **1.00×** | **178** | 100% |
| Swaram v2 | 0.564 | 2.926 | 1.02× | 175 | 100% |
| GPT-4o (`o200k`) | 0.667 | 3.463 | 1.21× | 148 | 100% |
| GPT-4 (`cl100k`) | 2.328 | 12.085 | **4.21×** | 42 | 100% |
| Code points | 1.808 | 9.383 | 3.26× | 55 | 100% |
| UTF-8 bytes | 4.740 | 24.604 | **8.56×** | 21 | 100% |

A Tamil passage that costs 100 tokens under Swaram costs **421** under GPT-4
and **856** as raw bytes. Under GPT-4, a 512-token window carries only **42**
Tamil words; Swaram carries **178** — a **4.2× larger effective context** for the
same window.

### 5.2 Tokenizer quality (300 held-out documents)

| artifact | vocab | fertility* | lossless | `<unk>` docs | throughput | morph. agreement |
|---|---|---|---|---|---|---|
| Swaram v2 (32k) | 31,964 | 0.339 | 100% | 0 | 2.48 Mchar/s | 5.6% |
| **Swaram v3 (32k, expanded)** | 31,964 | **0.334** | **100%** | **0** | **2.58 Mchar/s** | 5.6% |

\* Fertility here uses each artifact's own akshara layer on a different sample;
absolute values are therefore lower than §5.1. The meaningful comparison is
within a column.

### 5.3 Training efficiency

* Freezing the tokenizer on 300M characters: **172 s**.
* Packing 258.3M tokens at 512 context: **129 s** (≈2.0M tokens/s).
* Encoding throughput: **~2.5 Mchar/s** single process.

### 5.4 End-to-end validation

The v3 artifact is wired into the Adhan nano training path. A 40-step CPU run on
the packed shards reduced loss from **9.02 to 6.69** (val perplexity **795**),
confirming the tokenizer, shards, vocab and model all agree.

---

## 6. Why the akshara matters

**Compression.** Because merges operate on aksharas, a Tamil morpheme is one
token rather than a byte sequence. Fertility is near or below 1 token per
akshara, so the model spends capacity on meaning, not on re-assembling
characters.

**Robustness.** Segmentation is deterministic and lossless; byte fallback makes
the vocabulary total. There are no `<unk>` documents in evaluation.

**Morphological alignment (potential).** BPE over aksharas can learn whole
suffixes as tokens, which is exactly the structure Tamil needs. The current
artifact realises only part of this potential (see §7).

**Multilingual extension.** The same recipe applies to other akshara-based
scripts. Dravidian languages (Kannada, Telugu, Malayalam) share the
consonant+vowel-sign structure, so Layer A generalises with a script-specific
combining-mark range and inventory, and Layer B transfers with continued BPE
training. Devanagari adds conjunct clusters and needs an explicit conjunct
inventory.

---

## 7. Limitations and threats to validity

1. **Morphological boundary agreement is low (5.6%).** BPE merges often cross
   stem/suffix boundaries even though they never cross akshara boundaries. The
   `--morphology-guard` option is implemented but not yet evaluated at scale;
   this is the most important open quality question.
2. **Fertility is sample-sensitive.** Absolute fertility differs between the two
   benchmarks because they sample documents differently and count punctuation
   and Latin/digit runs in the numerator but not in the akshara denominator.
   Cross-tokenizer comparisons on the same sample (the tax) are the robust
   quantity.
3. **Corpus composition.** The corpus mixes classical, encyclopedic, news and
   web registers. A domain-specific deployment may see different fertility.
4. **Rust/Python parity.** The shipped fast path is Rust; the Python reference is
   used for tests. Parity is covered by tests but not formally proven for every
   Unicode edge case.
5. **No comparison to a Tamil-trained BPE baseline yet.** The tax is measured
   against general-purpose tokenizers; a BPE trained on Tamil bytes would be a
   stronger baseline and is planned.
6. **Single language.** Results are Tamil-only; the multilingual claims in §6 are
   design arguments, not measurements.

---

## 8. Reproducibility

All artifacts and scripts are in the Adhan repository.

```bash
# 1. Build the corpus (sources and library crawlers)
python scripts/expand_tamil_corpus_v2.py --resume --out-dir data/raw/expanded_v2
python scripts/ingest_tamilvu.py --out data/raw/web/tamilvu.jsonl
python scripts/ingest_wikisource_ta.py --out data/raw/web/wikisource_ta.jsonl

# 2. Freeze the tokenizer + pack shards
python scripts/freeze_fast_tokenizer.py \
  --corpus data/raw/expanded_v2 --out data/final/tamil_slm_v3 \
  --vocab-size 32000 --seq-len 512 --train-chars 300000000 --max-tokens 0

# 3. Benchmark the token tax
python scripts/benchmark_token_tax.py \
  --corpus data/raw/expanded_v2 --docs 400 \
  --swaram data/final/tamil_slm_v3 --swaram-v1 data/final/tamil_slm_v2 \
  --out docs/tokenizer/TOKEN_TAX_BENCHMARK_V3.md

# 4. Benchmark tokenizer quality
python scripts/benchmark_tokenizer_v2.py --corpus data/raw/expanded_v2 \
  --tokenizer data/final/tamil_slm_v2:swaram-v2-32k \
  --tokenizer data/final/tamil_slm_v3:swaram-v3-32k-expanded \
  --out docs/tokenizer/TOKENIZER_BENCHMARK_V3.md
```

Artifact provenance is recorded in `data/final/tamil_slm_v3/datasheet.json`
(vocab size, merges, inventory, corpus, code SHA, fertility, throughput,
pack time). Reports: `docs/tokenizer/TOKEN_TAX_BENCHMARK_V3.md`,
`docs/tokenizer/TOKENIZER_BENCHMARK_V3.md`.

---

## 9. Future work

* **Morphology-guarded merges** — evaluate `--morphology-guard` against the tax
  and boundary-agreement metrics; ship only if compression stays competitive.
* **Akshara-aware unigram/LM tokenizer** — compare BPE merges with a unigram
  model over aksharas for better morphological boundaries.
* **Dravidian family tokenizer** — shared inventory and merges for Tamil,
  Kannada, Telugu, Malayalam; measure cross-lingual transfer.
* **Rust Layer A as the only path** — retire the Python reference from
  production, keep it for conformance tests.
* **Downstream study** — does lower fertility improve sample efficiency,
  long-context recall, and sandhi correctness at equal compute? This is the
  central empirical question the whitepaper leaves open.
* **On-device budget** — measure tokenizer latency and memory inside the
  Illakiya IME and the Yazh Unity runtime.

---

## 10. Conclusion

The token tax is not an inherent property of Tamil; it is an artefact of
tokenizers designed for English. Swaram makes the akshara the atomic unit, and
on real Tamil text it removes **76%** of GPT-4's token overhead and **88%** of
the raw-byte overhead, while remaining fully lossless and fast enough to pack a
billion-character corpus in minutes.

The remaining work is quality, not compression: aligning merges with Tamil
morphology and proving downstream benefit. That is the next phase of Adhan.

---

## Appendix A — Glossary

| Term | Meaning |
|---|---|
| அக்ஷர / akshara | Tamil grapheme cluster: base + optional matra/pulli |
| உயிர் / uyir | vowel |
| மெய் / mei | consonant |
| உயிர்மெய் / uyirmei | consonant+vowel ligature |
| புள்ளி / pulli | virama, marks a pure consonant |
| புணர்ச்சி / puṉarcci | sandhi, word-boundary fusion |
| சுவரம் / swaram | the tokenizer's name; also vowel/sound |
| Fertility | tokens per akshara (lower is more compression) |
| Token tax | tokens_T / tokens_swaram for the same text |

## Appendix B — Calculating the tax

For a fixed text `x`:

```
tax_T = |T(x)| / |Swaram(x)|
```

Cost consequences for a corpus of `N` Swaram tokens:

| quantity | cost |
|---|---|
| training tokens | `tax_T × N` |
| inference tokens | `tax_T × N` |
| effective context | `window / tax_T` |
| decode steps | `tax_T × N` |

With `tax_GPT-4 = 4.21`, a 512-token window holds 178 Tamil words under Swaram
and 42 under GPT-4.

## Appendix C — Artifact card (v3)

| field | value |
|---|---|
| tokenizer | swaram (Rust backend) |
| script | Tamil |
| vocab size | 31,964 |
| merges | 28,549 |
| base inventory | 404 |
| byte fallback | yes |
| morphology guard | no |
| merge sample | 300M chars, stride 3 |
| corpus | `data/raw/expanded_v2` (538,562 docs) |
| packed tokens | 258,338,989 |
| train / val sequences | 338,758 / 6,878 |
| seq len | 512 |
| eval fertility | 0.5061 |
| eval lossless | 1.0 |
| encode throughput | 473.7 kchar/s (freeze harness) |
| code SHA | `dad6f47` |

---

*Adhan SLM — Yazhi. Tamil-first language technology.*

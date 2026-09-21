# Swaram Tokenizer Benchmark: vs BPE and SentencePiece

One-page comparison of the Swaram (akshara-native) tokenizer against a byte-level BPE tokenizer and a SentencePiece (BPE mode) tokenizer, trained and evaluated on the same frozen corpus.

## Methodology

- **Corpus:** Thirukkural (1330 documents: couplet + 3 classical
  commentaries each), fetched from `https://raw.githubusercontent.com/tk120404/thirukkural/master/thirukkural.json` via `src/data_scraper/raw_extractors/thirukkural_extractor.py`.
- **Split:** deterministic by document index -- every 10th document held out for eval (133 docs), the rest for training (1197 docs). No randomness, no seed needed.
- **Target vocab size:** 3000 for all three tokenizers. BPE's *actual* vocab (see table) can plateau below target -- the `tokenizers` library stops merging once no more frequent adjacent pairs exist in the training data, which can happen before the target on a small corpus. Swaram and SentencePiece both fill the full requested budget by falling back to lower-frequency merges/pieces, so they typically reach the target exactly.
- **Fertility** = tokens / akshara, using Swaram's own rule-based akshara segmenter (`segment_aksharas`) as the shared denominator for all three tokenizers -- akshara is the linguistically correct atomic unit for Tamil, so this keeps the comparison apples-to-apples regardless of how each tokenizer internally splits text.
- **Vocabulary coverage** = % of eval-set tokens that are NOT the tokenizer's `<unk>` token. Byte-level BPE has no `<unk>` by construction (any byte sequence is representable), so it always reports 100% here -- this is a real property of that approach, not a benchmark artifact. SentencePiece was trained with `byte_fallback=False` deliberately, so its coverage number reflects what its *learned* vocabulary actually covers, not a byte-level safety net.

## Results

| Tokenizer | Vocab size | Tokens/word | Vocab coverage | Fertility (mean) | Fertility (median) |
|---|---|---|---|---|---|
| Swaram | 3,000 | 2.768 | 99.99% | 0.606 | 0.605 |
| BPE (tokenizers, byte-level) | 1,678 | 6.338 | 100.00% | 1.388 | 1.391 |
| SentencePiece (BPE) | 3,000 | 2.053 | 99.89% | 0.450 | 0.441 |

## Interpretation

On this run, SentencePiece (BPE) achieves lower mean fertility (0.450 tokens/akshara) than Swaram (0.606). This is a genuine result, not a bug in the benchmark -- see the caveat below for the most likely reason, and re-running at a larger vocab size / on a less repetitive corpus is the natural next step to check whether it holds up.

A fertility near or below 1.0 means the tokenizer spends about one token (or fewer, once merges kick in) per akshara -- the theoretical floor for a lossless akshara-aware scheme -- rather than fragmenting Tamil script into multiple sub-akshara pieces the way byte-level tokenizers with no Tamil-specific prior tend to.

## Caveat: this corpus is unusually repetitive

Every Thirukkural document repeats the exact same boilerplate labels (`"மு. வரதராசனார் உரை:"`, `"சாலமன் பாப்பையா உரை:"`, `"கலைஞர் உரை:"`, `"திருக்குறள் <N>:"`) 1,330 times. A frequency-driven learner (BPE, SentencePiece) can dedicate a large slice of a *small* vocabulary budget to memorizing these exact recurring phrases as single efficient merges, which flatters tokens-per-word and fertility in a way that may not hold on more varied, less boilerplate-heavy Tamil text. Swaram's vocabulary is built akshara-first (247 base units) before merges, so a larger share of its budget goes to general coverage rather than corpus-specific memorization -- expected to matter more at larger vocab sizes and on more varied corpora, but not tested here. Treat this report as a first, reproducible data point, not a final verdict; a follow-up with a more varied corpus and multiple vocab sizes is the natural next step.

Numbers above are from a single run on the frozen split described in Methodology; re-running `scripts/benchmark_tokenizers.py` reproduces them exactly (same corpus source, same split rule, same vocab size).

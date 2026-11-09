# Swaram v4: beating Agaram's tokenizer on Tamil

**2026-11-09.** Agaram 2.5B had the best licence-clean Tamil tokenizer we had measured
(0.364 tokens per akshara). Swaram v4 reaches **0.307 at the same 64k vocabulary** —
16% fewer tokens per akshara, and it cuts 0.2% of Tamil aksharas instead of Agaram's 7.1%.
The frozen corpus, harness and sample rule are in
[`docs/TOKENIZER_BENCH.md`](../TOKENIZER_BENCH.md); every number below is from
[`tokenizer_bench_results.md`](tokenizer_bench_results.md).

## Result

1,500 sentences, 500 each of classical / modern / tanglish, seed 13.

| Tokenizer | vocab | Tokens/akshara | Tokens/word | Aksharas split |
|---|---:|---:|---:|---:|
| **swaram-v4-64k-mix** | 63,954 | **0.307** | 1.53 | **0.2%** |
| swaram-v4-96k-mix | 95,954 | 0.296 | 1.48 | 0.3% |
| agaram-2.5b | 64k byte BPE | 0.364 | 1.81 | 7.1% |
| sarvam-1 | — | 0.505 | 2.52 | 7.0% |
| swaram-v3 (shipped 32k) | 31,964 | 0.564 | 2.81 | 0.1% |
| sarvam-m | — | 0.602 | 3.00 | 30.1% |
| qwen3-1.7b / 4b | ~151k | 1.346 | 6.71 | 89.1% |

Per register, Swaram v4-64k-mix beats Agaram everywhere: classical 0.370 vs 0.404,
modern 0.318 vs 0.381, tanglish 0.252 vs 0.321. The 96k build is marginally better
(0.296) and is *not* the recommendation — see the embedding cost below.

## What actually moved the number

Three changes, in order of effect on the all-register figure (32k control → 64k → mix):

| Change | Tokens/akshara | Note |
|---|---:|---|
| v3 (32k, space as its own token, Tamil-only mix) | 0.564 | shipped artifact |
| v4-32k, corrected inventory, same convention | 0.580 | 15M chars of training data vs v3's 300M |
| + budget 32k → 64k | 0.546 | −6% |
| + Tanglish in the training mix | 0.507 | −7%, almost all of it in the tanglish register |
| **+ word mark merges with the following word** | **0.307** | −39% at 64k: the space stops costing a token |

The last line is the real bug. Swaram emitted `▁` as a standalone token for every space,
so every word cost one extra token (measured: 0.90 per word on the benchmark corpus).
SentencePiece-style tokenizers fold the space into the word that follows it (`▁மொழி`),
which is what Agaram does. Fixing it (`pre_tokenizers.Split(..., behavior="merged_with_next")`
in the Rust path, the same convention in `SwaramTokenizer.train`) removes ~0.9 tokens per
word and brings `tokens/word` from 2.52 to 1.53 — below Agaram's 1.81.

The Tanglish mixture matters because Swaram had almost no Latin merges: tanglish cost
0.508 without it, 0.252 with it. It costs 0.013 on classical and 0.007 on modern —
worth it for a tokenizer that will serve code-mixed Tamil input.

## Reproduce

```bash
# 1. local pools from the crawl + the CC-BY-4.0 Tanglish corpus (see TOKENIZER_BENCH.md)
python scripts/build_bench_sources.py

# 2. train the tokenizer (21M chars: 15M Tamil web + 6M Tanglish, 64k vocab, ~11s)
python scripts/train_tokenizer.py --vocab-size 64000 \
  --corpus data/raw/expanded --chars 15000000 \
  --corpus data/raw/tanglish/tanglish_corpus.jsonl --chars 6000000 \
  --out data/final/swaram_v4b_64k_mix

# 3. score it on the frozen corpus
python -m adhan_slm.eval.tokenizer_bench --corpus data/eval/tokenizer_bench.jsonl \
  --swaram data/final/swaram_v4b_64k_mix/vocab.json \
           data/final/swaram_v4b_64k_mix/merges.txt --swaram-name swaram-v4-64k-mix \
  --baselines --out-md docs/reports/tokenizer_bench_results.md
```

`data/` is gitignored, so the weights are not in this repository; the recipe travels in
`data/final/swaram_v4b_64k_mix/datasheet.json` (corpora, char budgets, inventory size,
merge count, seconds) and in the two commands above.

## Honest limits

- **Training data.** 21M characters. Agaram's tokenizer saw a corpus orders of magnitude
  larger, and v2/v3 saw 300M. More data should improve v4 further; this result is the
  floor, not the ceiling.
- **Vocabulary cost.** 64k tokens is 2× the v3 embedding matrix. At `d_model = 256`
  (`adhan-nano`) that is 16.4M parameters of embedding — larger than the model itself.
  For a nano-scale student, keep 32k (0.437) or 24k; ship 64k only with a ≥1B-token
  budget, where the embedding amortises. 96k is measured, not recommended.
- **Akshara splits.** 0.2% (v4) vs 0.1% (v3): the Tanglish mixture and raw Latin alphabet
  let a few merges cut a cluster. Still 35× better than Agaram's 7.1%, and it can be
  driven to zero by keeping the raw-alphabet merges out of Tamil (future work).
- **Not a fairness claim about corpora.** Both tokenizers were scored on the same frozen
  corpus, at the same 64k vocabulary. Agaram's *training* corpus is larger and unknown in
  detail; ours is the 21M chars above.
- **The v2/v3 artifacts are superseded** by the word-mark convention: their `vocab.json`
  ids do not carry `▁` merges, so a checkpoint trained on them cannot be re-tokenized into
  v4 ids. Retrain rather than remap.

<!--
DRAFT. Not for publication until the checklist in docs/BENCHMARK_RELEASE_CHECKLIST.md is done.
Every [[FILL: ...]] must be replaced with a number from docs/TOKENIZER_BENCH_RESULTS.md.
Do not publish any number that is not in that file.
Tamil text in this post must be checked by a native reviewer.
-->

# The Tamil token tax: what tokenizers cost on Tamil, and how to check it yourself

## The problem

Language models do not read words. They read tokens, and every tokenizer decides for itself
how to cut text into tokens. Tokenizers built mostly on English cut Tamil much more finely:
more tokens for the same sentence. That is the Tamil token tax. It is paid every time: in
training cost, in API cost, and in how much text fits in a context window.

There is a second cost that is easy to miss. Tamil is written in aksharas, a consonant with
its vowel sign (க + ா = கா). A tokenizer that cuts through an akshara produces pieces that
mean nothing on their own.

## What we measured

We built a small frozen corpus in three registers: classical Tamil, modern Tamil and Tanglish
(Tamil in Latin script). For each tokenizer we measure:

- **Fertility:** tokens per word and tokens per akshara.
- **Split aksharas:** the share of Tamil aksharas that more than one token touches.
- **Cost:** tokens and dollars per 1,000 sentences.

Corpus: [[FILL: number of sentences per register]]. Sources and licences are listed in the
manifest. Tokenizers compared: [[FILL: names and versions]]. Gemma and Llama are left out
for licence reasons. Everything ran locally; no text was sent to any API.

## Results

[[FILL: paste the overall table from docs/TOKENIZER_BENCH_RESULTS.md]]

What stands out:

- [[FILL: the largest tax, as a ratio of tokens against the best tokenizer, with the numbers]]
- [[FILL: how often Tamil aksharas are split for each tokenizer]]
- [[FILL: the register where the gap is largest or smallest]]

## What this means for Tamil

[[FILL: two or three sentences, only about what the table shows. Do not claim anything about
model quality; this benchmark measures tokens, not answers.]]

## Limits

- The corpus is small, so these are small-corpus numbers.
- Token counts are not model quality.
- Prices change; the cost column uses the price given in the config, [[FILL: price and date]].
- We compare only tokenizers we are allowed to run and redistribute results for.

## Reproduce it

```bash
git clone https://github.com/yazhi-lem/adhan && cd adhan
pip install -e ".[benchmark]"
scripts/reproduce_benchmark.sh --check
```

The script rebuilds the corpus, checks it against the committed manifest and compares every
number with the published tables. If anything differs, it fails and says where.

Code, corpus builder and tables: https://github.com/yazhi-lem/adhan
(folder `benchmarks/tamil-tokenizer`).

## Corrections and additions

If you find a mistake or want another tokenizer added, open an issue on the repository.

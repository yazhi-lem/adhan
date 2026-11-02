# Tamil tokenizer benchmark

How many tokens does a tokenizer spend on Tamil? This folder is the public entry point for a
benchmark that measures the "Tamil token tax" and lets anyone reproduce the numbers.

> Status: **draft, not yet published.** Results are filled in only after the frozen corpus is
> committed and `scripts/reproduce_benchmark.sh --check` passes from a clean checkout. See
> [the release checklist](../../docs/BENCHMARK_RELEASE_CHECKLIST.md).

## What it measures

On a frozen eval corpus in three registers (classical, modern, Tanglish):

| Metric | Meaning |
|---|---|
| Tokens/word, tokens/akshara | fertility: how finely the text is cut |
| Tamil clusters split | share of Tamil aksharas that more than one token touches |
| Tokens and USD per 1,000 sentences | what the tax costs in practice |

## Reproduce

```bash
git clone https://github.com/yazhi-lem/adhan && cd adhan
pip install -e ".[benchmark]"
scripts/reproduce_benchmark.sh --check
```

Every number comes from one committed script and one committed manifest. If a rebuilt corpus
differs from the manifest, the run fails instead of publishing different numbers.

## Where things are

| What | Where |
|---|---|
| Eval corpus builder and licence log | [`docs/EVAL_CORPUS.md`](../../docs/EVAL_CORPUS.md) |
| Harness and metric definitions | [`docs/TOKENIZER_BENCH.md`](../../docs/TOKENIZER_BENCH.md) |
| One-command reproduction | [`docs/REPRODUCE_BENCHMARK.md`](../../docs/REPRODUCE_BENCHMARK.md) |
| Results tables | `docs/TOKENIZER_BENCH_RESULTS.md` (added when published) |
| Blog post draft | [`docs/blog/tamil-token-tax.md`](../../docs/blog/tamil-token-tax.md) |

## Rules

- Tokenizers run locally. No corpus text is sent to any API.
- Only sources with a recorded licence are in the corpus. Unlicensed text is never committed.
- Gemma and Llama tokenizers are excluded for licence reasons.
- Tamil handling is NFC and grapheme-safe.

## Limits

The benchmark reports what it measures and nothing more: token counts and cluster splits, not
model quality. A small corpus gives small-corpus numbers.

<!--
DRAFT. Not for publication until the checklist in docs/BENCHMARK_RELEASE_CHECKLIST.md is done.
Every number below is copied from docs/TOKENIZER_BENCH_RESULTS.md (corpus sha256 24a66b9d…).
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
- **Cost:** tokens per 1,000 sentences, and dollars if a price is supplied in the config.

Corpus: **500 sentences per register, 1,500 in total** (classical: Thirukkural verses, public
domain; modern: Tamil Wikipedia, CC BY-SA; Tanglish: a CC-BY-4.0 social-media corpus). Sources
and licences are listed in the manifest. Tokenizers compared: **Swaram v3** (our akshara-native
tokenizer, 3,000 tokens), **Qwen3-1.7B-Base** (Apache-2.0, 151,665 tokens), and two
dependency-free baselines — **code point** and **UTF-8 byte** — so the table has a floor and a
ceiling. Gemma and Llama are left out for licence reasons. Everything ran locally; no text was
sent to any API.

## Results

All three registers together:

| Tokenizer | Tokens/word | Tokens/akshara | Tamil aksharas split | Tokens/1k sentences |
|---|---:|---:|---:|---:|
| Swaram v3 | 4.86 | 0.957 | 0.0% | 39,944 |
| Qwen3-1.7B-Base | 6.16 | 1.215 | 89.1% | 50,707 |
| code point | 7.65 | 1.509 | 70.5% | 62,941 |
| UTF-8 byte | 15.88 | 3.132 | 100.0% | 130,683 |

What stands out:

- **The tax is 1.27×.** Qwen3 spends 1.215 tokens per akshara where Swaram spends 0.957, and
  50,707 tokens per 1,000 sentences where Swaram spends 39,944. Same text, 27% more tokens.
- **Qwen3 cuts through 89.1% of Tamil aksharas.** Swaram cuts through none, by construction;
  the code-point baseline cuts 70.5% and raw UTF-8 bytes cut all of them. So Qwen3 is *better*
  than a naive code-point split (0.81× its tokens) — it does learn Tamil merges — but it learns
  them without respecting akshara boundaries.
- **The register changes the answer completely.** On classical Tamil Qwen3 costs 3.08× Swaram
  (2.118 vs 0.687 tokens/akshara); on modern Tamil 2.54× (1.978 vs 0.778). On **Tanglish it
  inverts**: Qwen3 costs 0.374 against Swaram's 1.170, because Tanglish is Latin script and
  Swaram is built for Tamil script. A tokenizer choice that is right for one register is wrong
  for another.

## What this means for Tamil

On Tamil script, an akshara-native tokenizer spends about a quarter fewer tokens than a
general-purpose one, and it never splits an akshara. On Tamil written in Latin script the
advantage disappears and reverses. So "which tokenizer is better for Tamil" has no single
answer: it depends on which Tamil you are serving, and a product that serves both needs to
know that before it picks one.

## Limits

- The corpus is small (1,500 sentences), so these are small-corpus numbers.
- Token counts are not model quality. A tokenizer that spends fewer tokens is not automatically
  a better model; this benchmark measures tokens, not answers.
- No dollar column: no price is configured, so cost is reported in tokens only. Add
  `price_per_million` to `eval_manifests/tokenizer_bench.reproduce.json` to get USD per 1,000
  sentences.
- We compare only tokenizers we are allowed to run and redistribute results for.

## Reproduce it

```bash
git clone https://github.com/yazhi-lem/adhan && cd adhan
pip install -e ".[benchmark]"
scripts/reproduce_benchmark.sh --check
```

The script rebuilds the corpus, checks it against the committed manifest and compares every
number with the published tables. If anything differs, it fails and says where. The Swaram
tokenizer files are committed under `eval_manifests/tokenizers/swaram-v3/`, and the Qwen3
tokenizer is fetched from a pinned URL, so a clean clone needs no manual setup.

Code, corpus builder and tables: https://github.com/yazhi-lem/adhan
(folder `benchmarks/tamil-tokenizer`).

## Corrections and additions

If you find a mistake or want another tokenizer added, open an issue on the repository.

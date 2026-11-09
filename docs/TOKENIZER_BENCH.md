# Tamil tokenizer benchmark harness

Part 2 of [#74](https://github.com/yazhi-lem/adhan/issues/74). It measures what a tokenizer
costs on the frozen eval corpus built in part 1 ([EVAL_CORPUS.md](EVAL_CORPUS.md)).

## Metrics

| Metric | Meaning |
|---|---|
| Tokens/word | tokens divided by whitespace-separated words |
| Tokens/akshara | tokens divided by Tamil grapheme clusters (`segment_aksharas`) |
| Tamil clusters split | share of Tamil aksharas that more than one token touches. A cluster is split when a token boundary cuts it (`க` + `ா`) or when a character is spread over several byte tokens |
| Tokens/1k sentences | `tokens / sentences * 1000` |
| USD/1k sentences | tokens/1k sentences times `--price-per-million` / 1e6 (only with a price) |

All metrics are reported overall and per register (classical, modern, tanglish). The split
rate is `n/a` for a register with no Tamil script. Text is NFC-normalised before tokenizing.

## Supported tokenizers

- Swaram: `--swaram vocab.json merges.txt` (repeatable; label each with `--swaram-name`,
  otherwise rows are `swaram` or `swaram-N`). Byte fallback is counted token by token
- Any `tokenizers` tokenizer: `--hf path/to/tokenizer.json`
- Any `transformers` fast tokenizer in a local directory: `--hf path/to/dir`
  (downloads only with `--allow-hub`, and only tokenizer files, never your text)
- Baselines: `--baselines` adds one-token-per-code-point and one-token-per-UTF-8-byte

Gemma and Llama tokenizers are excluded for licence reasons.

## Run

```bash
PYTHONPATH=src python -m adhan_slm.eval.tokenizer_bench \
  --corpus data/eval/tokenizer_bench.jsonl \
  --swaram vocab.json merges.txt \
  --hf path/to/other/tokenizer.json \
  --baselines --price-per-million 0.5 \
  --out-json results.json --out-md results.md
```

Everything runs locally. The harness never sends text anywhere.

## Notes

- A token needs character offsets to be checked for splits. Fast Hugging Face tokenizers
  provide them; slow tokenizers do not and are rejected with a clear error.
- Tiktoken is not supported yet (no character offsets); it is a follow-up.
- Part 3 will wrap corpus build, harness and tables into one reproduce command.

## Frozen corpus (2026-11-09) and results

The real spec is [`eval_manifests/tokenizer_bench.sources.json`](../eval_manifests/tokenizer_bench.sources.json):
four licensed sources, 500 sentences per register, seed 13, ranked by
`sha256(seed|source|text)`.

| Source | Register | Licence | Pool |
|---|---|---|---|
| Thirukkural (classical) | classical | Public domain | 1,330 docs / 665k chars |
| Kalithokai (Sangam) | classical | Public domain | 8,000 docs / 2.0M chars |
| Tamil Wikipedia | modern | CC BY-SA 4.0 | 8,000 docs / 9.8M chars |
| Tanglish-Corpus-185k @ `927f5b4` | tanglish | CC-BY-4.0 | 185,973 docs / 27.9M chars |

The two crawl-derived pools are cut from `data/raw/expanded` by
`scripts/build_bench_sources.py` (label filter, 40–6,000 chars, exact-dup removal,
first 8,000 — the rule is recorded in each pool's `.manifest.json`). The Tanglish file
is fetched from the Hub at the pinned revision. Raw text stays under `data/` (gitignored);
only [`tokenizer_bench.manifest.json`](../eval_manifests/tokenizer_bench.manifest.json)
(per-source counts, sentence counts, SHA-256) and the results tables are committed.

Rebuild:

```bash
python scripts/build_bench_sources.py                       # local pools from the crawl
python scripts/build_eval_corpus.py \
  --spec eval_manifests/tokenizer_bench.sources.json        # data/eval/tokenizer_bench.jsonl
python -m adhan_slm.eval.tokenizer_bench \
  --corpus data/eval/tokenizer_bench.jsonl \
  --swaram data/final/tamil_slm_v2/vocab.json data/final/tamil_slm_v2/merges.txt \
  --swaram-name swaram-v2 \
  --hf data/eval/tokenizers/qwen3-1.7b.json --hf data/eval/tokenizers/sarvam-m.json \
  --baselines --out-json docs/reports/tokenizer_bench_results.json \
  --out-md docs/reports/tokenizer_bench_results.md
```

Results (1,500 sentences, all registers): [tokenizer_bench_results.md](reports/tokenizer_bench_results.md).
Headline numbers, tokens per akshara — the lower the better:

| Tokenizer | all | classical | modern | tanglish | Tamil clusters split |
|---|---:|---:|---:|---:|---:|
| agaram-2.5b | **0.364** | 0.404 | 0.381 | 0.321 | 7.1% |
| sarvam-1 | 0.505 | 0.543 | 0.526 | 0.459 | 7.0% |
| swaram-v3 | 0.564 | 0.525 | 0.522 | 0.627 | 0.1% |
| swaram-v2 | 0.595 | 0.518 | 0.525 | 0.627 | 0.3% |
| sarvam-m | 0.602 | 0.729 | 0.765 | 0.373 | 30.1% |
| qwen3-1.7b / 4b | 1.346 | 1.952 | 1.975 | 0.375 | 89.1% |
| codepoint | 1.562 | 1.808 | 1.811 | 1.173 | 68.5% |
| utf8-byte | 3.436 | 4.870 | 4.885 | 1.173 | 100.0% |

Two facts worth carrying forward: Qwen3-1.7B and Qwen3-4B **share one tokenizer**, so
moving to 4B buys no token efficiency; and Swaram is the only tokenizer that never cuts
a Tamil akshara (0.1–0.3%), which is the point of the akshara layer.

These numbers are not comparable to `bases_tokenizer_bench.json` (web-only, 300 docs,
seed `adhan-bases-2026-11-05`): Qwen3 measures 2.027 tok/akshara there and 1.346 here
because this corpus contains Tanglish, where a BPE is nearly one token per word. Always
quote the sample rule with the number.

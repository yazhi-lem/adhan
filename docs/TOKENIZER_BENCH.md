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

- Swaram: `--swaram vocab.json merges.txt` (byte fallback is counted token by token)
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

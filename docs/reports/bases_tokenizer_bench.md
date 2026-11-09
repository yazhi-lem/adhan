# Candidate base tokenizers on one Tamil sample

- sample: `data/raw/expanded/corpus-00001.jsonl`, **300 docs**, seed `adhan-bases-2026-11-05`
- size: **59,699 characters · 38,991 aksharas**
- harness: `src/adhan_slm/eval/tokenizer_bench.py` (NFC, grapheme-safe offsets)

# Tamil tokenizer benchmark

## All registers

| Tokenizer | Tokens/word | Tokens/akshara | Tamil clusters split | Tokens/1k sent. |
|---|---:|---:|---:|---:|
| agaram-2.5b | 1.88 | 0.418 | 8.5% | 44,673 |
| qwen3-1.7b | 9.13 | 2.027 | 89.5% | 216,777 |
| sarvam-1 | 2.63 | 0.584 | 7.7% | 62,423 |
| sarvam-m | 3.58 | 0.795 | 31.6% | 85,047 |
| swaram-v3 | 2.62 | 0.582 | 0.0% | 62,217 |
| swaram-v2 | 2.67 | 0.593 | 0.3% | 63,377 |

## modern

| Tokenizer | Tokens/word | Tokens/akshara | Tamil clusters split | Tokens/1k sent. |
|---|---:|---:|---:|---:|
| agaram-2.5b | 1.88 | 0.418 | 8.5% | 44,673 |
| qwen3-1.7b | 9.13 | 2.027 | 89.5% | 216,777 |
| sarvam-1 | 2.63 | 0.584 | 7.7% | 62,423 |
| sarvam-m | 3.58 | 0.795 | 31.6% | 85,047 |
| swaram-v3 | 2.62 | 0.582 | 0.0% | 62,217 |
| swaram-v2 | 2.67 | 0.593 | 0.3% | 63,377 |

Lower is better for every column. 'Tamil clusters split' is the share of Tamil aksharas that more than one token touches (n/a when a register has no Tamil script).

# Tamil tokenizer benchmark

## All registers

| Tokenizer | Tokens/word | Tokens/akshara | Tamil clusters split | Tokens/1k sent. |
|---|---:|---:|---:|---:|
| swaram-v4-64k | 1.42 | 0.286 | 0.0% | 14,593 |
| swaram-v4-32k | 1.58 | 0.317 | 0.0% | 16,221 |
| swaram-v3 | 2.81 | 0.564 | 0.1% | 28,819 |
| swaram-v2 | 2.96 | 0.595 | 0.3% | 30,374 |
| agaram-2.5b | 1.81 | 0.364 | 7.1% | 18,583 |
| sarvam-1 | 2.52 | 0.505 | 7.0% | 25,789 |
| sarvam-m | 3.00 | 0.602 | 30.1% | 30,781 |
| qwen3-1.7b | 6.71 | 1.346 | 89.1% | 68,765 |
| qwen3-4b | 6.71 | 1.346 | 89.1% | 68,765 |
| codepoint | 7.78 | 1.562 | 68.5% | 79,799 |
| utf8-byte | 17.12 | 3.436 | 100.0% | 175,565 |

## classical

| Tokenizer | Tokens/word | Tokens/akshara | Tamil clusters split | Tokens/1k sent. |
|---|---:|---:|---:|---:|
| swaram-v4-64k | 1.37 | 0.290 | 0.0% | 12,094 |
| swaram-v4-32k | 1.50 | 0.319 | 0.0% | 13,306 |
| swaram-v3 | 2.47 | 0.525 | 0.0% | 21,890 |
| swaram-v2 | 2.44 | 0.518 | 0.0% | 21,586 |
| agaram-2.5b | 1.90 | 0.404 | 7.4% | 16,842 |
| sarvam-1 | 2.56 | 0.543 | 7.0% | 22,636 |
| sarvam-m | 3.43 | 0.729 | 27.1% | 30,356 |
| qwen3-1.7b | 9.19 | 1.952 | 89.6% | 81,306 |
| qwen3-4b | 9.19 | 1.952 | 89.6% | 81,306 |
| codepoint | 8.51 | 1.808 | 68.0% | 75,330 |
| utf8-byte | 22.92 | 4.870 | 100.0% | 202,906 |

## modern

| Tokenizer | Tokens/word | Tokens/akshara | Tamil clusters split | Tokens/1k sent. |
|---|---:|---:|---:|---:|
| swaram-v4-64k | 1.50 | 0.306 | 0.0% | 15,902 |
| swaram-v4-32k | 1.68 | 0.342 | 0.0% | 17,782 |
| swaram-v3 | 2.56 | 0.522 | 0.2% | 27,142 |
| swaram-v2 | 2.57 | 0.525 | 0.5% | 27,268 |
| agaram-2.5b | 1.86 | 0.381 | 6.9% | 19,786 |
| sarvam-1 | 2.57 | 0.526 | 7.0% | 27,326 |
| sarvam-m | 3.74 | 0.765 | 32.5% | 39,738 |
| qwen3-1.7b | 9.67 | 1.975 | 88.7% | 102,618 |
| qwen3-4b | 9.67 | 1.975 | 88.7% | 102,618 |
| codepoint | 8.87 | 1.811 | 68.9% | 94,118 |
| utf8-byte | 23.92 | 4.885 | 100.0% | 253,840 |

## tanglish

| Tokenizer | Tokens/word | Tokens/akshara | Tamil clusters split | Tokens/1k sent. |
|---|---:|---:|---:|---:|
| swaram-v4-64k | 1.40 | 0.265 | n/a | 15,782 |
| swaram-v4-32k | 1.56 | 0.295 | n/a | 17,576 |
| swaram-v3 | 3.31 | 0.627 | n/a | 37,426 |
| swaram-v2 | 3.74 | 0.709 | n/a | 42,268 |
| agaram-2.5b | 1.69 | 0.321 | n/a | 19,122 |
| sarvam-1 | 2.43 | 0.459 | n/a | 27,406 |
| sarvam-m | 1.97 | 0.373 | n/a | 22,248 |
| qwen3-1.7b | 1.98 | 0.375 | n/a | 22,370 |
| qwen3-4b | 1.98 | 0.375 | n/a | 22,370 |
| codepoint | 6.19 | 1.173 | n/a | 69,948 |
| utf8-byte | 6.19 | 1.173 | n/a | 69,948 |

Lower is better for every column. 'Tamil clusters split' is the share of Tamil aksharas that more than one token touches (n/a when a register has no Tamil script).

# Tamil tokenizer benchmark

## All registers

| Tokenizer | Tokens/word | Tokens/akshara | Tamil clusters split | Tokens/1k sent. |
|---|---:|---:|---:|---:|
| swaram-v3 | 4.86 | 0.957 | 0.0% | 39,944 |
| qwen3-1.7b-base | 6.16 | 1.215 | 89.1% | 50,707 |
| codepoint | 7.65 | 1.509 | 70.5% | 62,941 |
| utf8-byte | 15.88 | 3.132 | 100.0% | 130,683 |

## classical

| Tokenizer | Tokens/word | Tokens/akshara | Tamil clusters split | Tokens/1k sent. |
|---|---:|---:|---:|---:|
| swaram-v3 | 3.12 | 0.687 | 0.0% | 10,874 |
| qwen3-1.7b-base | 9.63 | 2.118 | 90.1% | 33,548 |
| codepoint | 8.51 | 1.872 | 73.9% | 29,642 |
| utf8-byte | 23.80 | 5.237 | 100.0% | 82,930 |

## modern

| Tokenizer | Tokens/word | Tokens/akshara | Tamil clusters split | Tokens/1k sent. |
|---|---:|---:|---:|---:|
| swaram-v3 | 3.86 | 0.778 | 0.0% | 37,656 |
| qwen3-1.7b-base | 9.81 | 1.978 | 88.8% | 95,756 |
| codepoint | 8.99 | 1.814 | 69.3% | 87,814 |
| utf8-byte | 24.34 | 4.912 | 100.0% | 237,750 |

## tanglish

| Tokenizer | Tokens/word | Tokens/akshara | Tamil clusters split | Tokens/1k sent. |
|---|---:|---:|---:|---:|
| swaram-v3 | 6.24 | 1.170 | n/a | 71,302 |
| qwen3-1.7b-base | 2.00 | 0.374 | n/a | 22,816 |
| codepoint | 6.24 | 1.171 | n/a | 71,368 |
| utf8-byte | 6.24 | 1.171 | n/a | 71,368 |

Lower is better for every column. 'Tamil clusters split' is the share of Tamil aksharas that more than one token touches (n/a when a register has no Tamil script).

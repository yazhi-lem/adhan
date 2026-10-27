# Swaram vocabulary sweep

Frozen Thirukkural corpus: 1197 training docs, 133 held-out docs.
Lower tokens/akshara is better (pilot target < 1.15).

| Vocab | Byte fallback | Actual | Tokens/akshara | Tokens/word | `<unk>` rate | Round-trip | Train time |
|---:|:---:|---:|---:|---:|---:|---:|---:|
| 1000 | no | 1000 | 0.542 | 2.47 | 0.01% | 98.5% | 0.7s |
| 1000 | yes | 1000 | 0.611 | 2.79 | 0.00% | 100.0% | 0.6s |
| 2000 | no | 2000 | 0.445 | 2.03 | 0.01% | 98.5% | 0.9s |
| 2000 | yes | 2000 | 0.460 | 2.10 | 0.00% | 100.0% | 0.9s |
| 4000 | no | 4000 | 0.386 | 1.76 | 0.01% | 98.5% | 1.1s |
| 4000 | yes | 4000 | 0.391 | 1.78 | 0.00% | 100.0% | 1.1s |
| 8000 | no | 8000 | 0.344 | 1.57 | 0.01% | 98.5% | 1.2s |
| 8000 | yes | 8000 | 0.345 | 1.58 | 0.00% | 100.0% | 1.2s |
| 12000 | no | 12000 | 0.328 | 1.50 | 0.01% | 98.5% | 1.3s |
| 12000 | yes | 12000 | 0.329 | 1.50 | 0.00% | 100.0% | 1.3s |

Round-trip is `decode(encode(text)) == NFC(text)` on the held-out split.

## How to read this

Swaram emits one token per akshara, so tokens/akshara below 1 is really a
*merge rate*: even the 1k vocabulary clears the 1.15 target. The number is
therefore a mechanism check, not a discriminating gate. The comparison that
matters is against other tokenizers measured on the same sample:

| Tokenizer | Tokens/akshara | Role |
|---|---:|---|
| Qwen3-1.7B-Base | 2.016 | product student |
| Sarvam-1 | 0.538 | licence-rejected |
| Sarvam-M 24B | 0.804 | teacher |
| Tamil-trained BPE | 0.346 | research baseline |

Swaram only approaches the Tamil-trained BPE at 12k, and on this corpus that
gain is mostly memorisation, not generalisation. The compression thesis is
recorded as unsupported on the larger A/B
([`ab_swaram_vs_bpe.md`](ab_swaram_vs_bpe.md)).

Byte fallback is the honest lossless switch: +100% round-trip at every size,
and from 2k up it costs almost nothing in compression. At 1k it costs ~13%
because its 256 byte tokens occupy slots that would otherwise hold merges.

Caveat: the corpus is small, so large vocabularies mostly memorise whole words
from the training split. Treat this as a check of the mechanism and trainer
speed, not a final 12k validation, which needs a larger corpus.

Regenerate: `python scripts/swaram_vocab_sweep.py`

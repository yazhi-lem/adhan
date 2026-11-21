# Swaram v3 — frozen tokenizer for the public benchmark

Committed so the published table can be reproduced from a clean clone: the tokenizer is the
thing being measured, so it cannot live in git-ignored `data/`.

| File | SHA-256 |
|---|---|
| `vocab.json` | `7b5cfdffdfa630a8cee668e316eaf5ae1105c8c25eac231fa6f4bd852af73fb1` |
| `merges.txt` | `458ab4630c586eb76ff077ecf4c1870acf0e9de977e81d2b275e60f06a394fa3` |

Vocab size: **3,000** tokens · merges: **2,887**

Produced by `scripts/prepare_slm_corpus.py` (Swaram akshara BPE with byte fallback) and
frozen; the benchmark measures this artifact, it does not retrain it. Regenerating it would
change the numbers, so treat these files as immutable and add a new version directory instead.

Used by `eval_manifests/tokenizer_bench.reproduce.json`.

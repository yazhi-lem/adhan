# Swaram v4: beating Agaram's tokenizer on Tamil, without splitting an akshara

**2026-11-09.** Agaram 2.5B had the best licence-clean Tamil tokenizer we had measured
(0.364 tokens per akshara). Swaram v4 reaches **0.286 at the same 64k vocabulary**, and
the 32k build that a nano-scale model can actually afford reaches **0.317** — still
better than Agaram, at half the embedding cost. Neither build cuts a single Tamil akshara
into two tokens (0.0% of 42,611 aksharas) against Agaram's 7.1%.

Numbers come from the frozen 1,500-sentence corpus (500 each of classical / modern /
tanglish, seed 13): [`tokenizer_bench_results.md`](tokenizer_bench_results.md). Sample
rule and rebuild steps: [`docs/TOKENIZER_BENCH.md`](../TOKENIZER_BENCH.md).

## Result

| Tokenizer | vocab | Tokens/akshara | Tokens/word | Aksharas split | Tokens/1k sent. |
|---|---:|---:|---:|---:|---:|
| **swaram-v4-64k** | 63,959 | **0.286** | 1.42 | **0.0%** | 14,593 |
| **swaram-v4-32k** | 31,959 | **0.317** | 1.58 | **0.0%** | 16,221 |
| agaram-2.5b | 64k byte BPE | 0.364 | 1.81 | 7.1% | 18,583 |
| sarvam-1 | — | 0.505 | 2.52 | 7.0% | 25,789 |
| swaram-v3 (shipped) | 31,964 | 0.564 | 2.81 | 0.1% | 28,819 |
| sarvam-m | — | 0.602 | 3.00 | 30.1% | 30,781 |
| qwen3-1.7b / 4b | ~151k | 1.346 | 6.71 | 89.1% | 68,765 |

Per register, both v4 builds win everywhere:

| Register | swaram-v4-64k | swaram-v4-32k | agaram-2.5b |
|---|---:|---:|---:|
| classical | 0.290 | 0.319 | 0.404 |
| modern | 0.306 | 0.342 | 0.381 |
| tanglish | 0.265 | 0.295 | 0.321 |

## What moved the number

| Step | Tokens/akshara | Splits |
|---|---:|---:|
| v3 (32k, 300M chars, space as its own token, Tamil-only mix) | 0.564 | 0.1% |
| v4-32k / v4-64k, corrected inventory, space still a token | 0.580 / 0.546 | — |
| + Tanglish in the training mix (21M chars) | 0.507 | 0.2% |
| **+ the word mark merges with the following word** | **0.307** | 0.2% |
| **+ 70M training chars, whitespace-collapsed, data-driven inventory** | **0.286** (64k), 0.317 (32k) | **0.0%** |

Three fixes, in order of effect:

1. **`▁` merges with the following word** (SentencePiece convention). Swaram emitted the
   word mark as a standalone token, so every space cost a token — 0.90 per word measured.
   `pre_tokenizers.Split(..., behavior="merged_with_next")` in the Rust path and the same
   convention in `SwaramTokenizer.train`. This alone took 0.507 → 0.307.
2. **Tanglish in the mixture.** Tamil-only training left the merge table almost without
   Latin: tanglish cost 0.508. With 6–10M chars of the CC-BY-4.0 Tanglish corpus it is
   0.265–0.295, at a cost of ~0.01 on classical and modern.
3. **Whitespace collapse during training + a data-driven inventory.** Scraped documents
   carry `\n`/`\t`; those became merge pairs, and because `merges.txt` is one
   tab-separated merge per line, a newline merge **silently corrupted the file** and
   re-ranked the Python path (101 of 1,500 sentences tokenized differently from the Rust
   path before this). Training now collapses whitespace, and aksharas the table does not
   know but that occur ≥20 times are added to the inventory so a learned merge cannot cut
   them. Runtime text keeps its whitespace and stays byte-lossless; only training is
   normalised. Splits went 0.2% → 0.0% and Python/Rust parity went 101/1,500 → 1/1,500.

## Reproduce

```bash
python scripts/build_bench_sources.py                     # local pools (crawl + Tanglish)

# 70M chars: 60M Tamil web + 10M Tanglish, ~2m45s each including sampling
python scripts/train_tokenizer.py --vocab-size 64000 --extend-inventory-min-count 20 \
  --corpus data/raw/expanded --chars 60000000 \
  --corpus data/raw/tanglish/tanglish_corpus.jsonl --chars 10000000 \
  --out data/final/swaram_v4e_64k_big

python -m adhan_slm.eval.tokenizer_bench --corpus data/eval/tokenizer_bench.jsonl \
  --swaram data/final/swaram_v4e_64k_big/vocab.json \
           data/final/swaram_v4e_64k_big/merges.txt --swaram-name swaram-v4-64k \
  --baselines --out-md docs/reports/tokenizer_bench_results.md
```

`data/` is gitignored, so the weights are not in this repository; the recipe travels in
`data/final/swaram_v4e_64k_big/datasheet.json` (corpora, budgets, inventory size, merge
count, seconds) and in the two commands above.

## Honest limits

- **Training data volume.** 70M characters. Agaram's tokenizer saw a corpus orders of
  magnitude larger, and v2/v3 saw 300M Tamil-only. More data should improve v4 further;
  this is a floor, not a ceiling.
- **One known path difference.** 1 of 1,500 benchmark sentences tokenizes differently
  between the Rust and Python paths. The cause is a ZWJ inside a word (`இன்னு‍`): the
  Python path receives the ZWJ cluster from the segmenter as one unit, the Rust path has
  no composition merge for it and merges around it. Ids still round-trip; it is a
  documented difference, not a silent bug.
- **Fairness.** Both tokenizers are scored on the same frozen corpus at the same 64k
  vocabulary. Agaram's *training* corpus is larger and its exact composition is unknown.
- **v2/v3 are superseded.** The word-mark convention changes the token alphabet, so their
  vocabularies cannot be remapped; retrain rather than remap.
- **Splits are a metric, not a proof.** 0.0% is the measured share of the 42,611 Tamil
  aksharas in this corpus that more than one token touches. A different corpus can
  surface a cluster the inventory still lacks; `--extend-inventory-min-count` is the knob.

# Adhan Corpus Expansion V2 — toward 1.5–2B clean swaram tokens

`scripts/expand_tamil_corpus_v2.py` extends the v1 archive (`docs/corpus/CORPUS_EXPANSION.md`). It adds sources, raises
limits, and dedups across versions so that v2 never re-emits a v1 document.

## Result so far (Phase 2 totals, 27 Oct 2026)

| | Docs | Swaram tokens |
|---|---:|---:|
| v1 archive | 1,334,126 | ~328M |
| v2 archive (new, deduped vs v1) | 538,562 | ~281.7M |
| **Combined** | **~1.87M** | **~609.7M** (≈ 35% of the 1.5–2B target) |

Landed: mC4 198,864 · IndicCorp 197,082 · FineWeb-2 42,150 · Tamil Virtual Academy 976 · Wikisource 77 · local
classical · Wikipedia. **Not available:** CC-100 (script datasets no longer supported), OSCAR (gated, needs
`HF_TOKEN`), news/books (0 accepted). To grow: raise `--indic-limit`, `--mc4-limit` and `--fineweb-limit`, then
resume. CC-100 and OSCAR need a converted mirror or a token.

## Run

```bash
python scripts/expand_tamil_corpus_v2.py --full --out-dir data/raw/expanded_v2              # all sources, 4–8 h
python scripts/expand_tamil_corpus_v2.py --seed-v1 data/raw/expanded --mc4-limit 2000000 \
    --out-dir data/raw/expanded_v2                                                          # incremental
python scripts/ingest_tamilvu.py … && python scripts/ingest_wikisource_ta.py …              # libraries (docs/corpus/DIGITAL_LIBRARIES.md)
python scripts/expand_tamil_corpus_v2.py --recount-only --out-dir data/raw/expanded_v2      # cumulative stats
```

The runner is resumable (`state.json`) and isolates per-source failures, so a gated dataset can't abort the build.
Sources are revision-pinned with `ADHAN_HF_REVISION_*`.

## Quality gates

| Source | `min_chars` | `min_tamil_ratio` |
|---|---:|---:|
| Local classical | 30 | 0.50 |
| Wikipedia | 80 | 0.55 |
| IndicCorp v2 | 50 | 0.60 |
| FineWeb-2 (also language score ≥ 0.90) | 60 | 0.60 |
| mC4 | 80 | 0.50 |
| News / books / web scrape | 100 / 200 / 40 | 0.60 / 0.55 / 0.50 |

**Dedup:** SHA-1 exact match, plus banded LSH at Jaccard 0.82 (stricter than v1's 0.80 for the noisier web sources),
seeded with all of v1.

## After expansion

Retrain the tokenizer (`freeze_fast_tokenizer.py --corpus data/raw/expanded_v2 --vocab-size 32000`) and re-run
`benchmark_token_tax.py`; a retrain that raises the tax is rejected. Pack shards, and fix cross-document packing first
(`factory/NEXT.md` N7). Budget: ~90 GB disk and ~16 GB RAM peak for expansion + tokenizer + packing.

## Later

2B → 3B: licensed transcripts and subtitles. 5B: open-data partnerships. 10B: a custom CommonCrawl Tamil extraction.

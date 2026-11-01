# Corpus Expansion Pipeline

`scripts/expand_tamil_corpus.py` streams public Tamil corpora, applies hygiene +
quality gates, removes exact and near duplicates, and writes sharded JSONL under
`data/raw/expanded/`. It is designed to be **re-run and extended**: each run
appends new shards without duplicating what is already on disk.

This document describes the sources, the resume semantics, and the memory
constraints that matter at the 1M-document scale.

---

## Sources

| Source | CLI flag | Default limit | Tier | Register |
|---|---|---|---|---|
| Local Sangam / classical roots | `--local-roots` | all found | 1–2 | Classical poetry, curated |
| Tamil Wikipedia (HF dataset) | `--wiki-limit` | 100,000 | 2 | Encyclopaedic |
| AI4Bharat IndicCorpV2 `tam_Taml` | `--indic-limit` | 200,000 | 3 | News / books / web |
| HuggingFace FineWeb-2 `tam_Taml` | `--fineweb-limit` | 0 (off) | 3 | Forum / blog / consumer web |

All remote sources are streamed (`load_dataset(..., streaming=True)`), so nothing
is downloaded in full before filtering.

### FineWeb-2

FineWeb-2 supplies a per-document `language_score`, which is a much stronger gate
than the character-ratio heuristic used for other sources. Documents below
`--fineweb-min-lang-score` (default `0.90`) are discarded *before* the expensive
quality pass, and the score is carried into the record as `quality_score`.

Only the `tam_Taml` config is used; `tam_Latn` (romanised/Tanglish) is excluded.

### Quality gates

`is_quality()` rejects documents that are too short (`min_chars`), have too low a
Tamil-script ratio (`min_tamil_ratio`), or are dominated by punctuation/digits.
Thresholds differ per source (local corpora are allowed to be shorter).

---

## Usage

```bash
# Fresh archive from local + Wikipedia + IndicCorp
python scripts/expand_tamil_corpus.py --out-dir data/raw/expanded \
    --wiki-limit 100000 --indic-limit 200000

# Extend an existing archive with a FineWeb-2 slice
python scripts/expand_tamil_corpus.py --out-dir data/raw/expanded \
    --wiki-limit 0 --indic-limit 0 --fineweb-limit 300000

# Local-only pass (no network)
python scripts/expand_tamil_corpus.py --local-only
```

Useful flags:

- `--shard-size` — documents per JSONL shard (default 20,000).
- `--no-local` — skip the on-disk local corpora.
- `--no-seed-existing` — skip preloading existing shards (fresh out-dir only).
- `--reset` — start over.

Progress is logged every 5,000 kept documents and after every shard rotation.
Completed remote sources are recorded in `data/raw/expanded/state.json` and
skipped on subsequent runs (`state.json` also stores the last kept count and
shard index).

---

## Resume and extension semantics

Two separate mechanisms keep re-runs idempotent:

1. **Seeding (start of run).** Every existing shard is read and its documents are
   registered by SHA-1 in `LshDedup.seed_exact()`. A re-run therefore drops
   documents that a previous run already wrote — important because streaming
   sources restart from the beginning, so raising `--indic-limit` re-encounters
   all previously consumed rows.
2. **`state.json`.** Finished sources are marked done and not streamed again.
   To deliberately re-stream a source with a larger limit, remove it from
   `done_sources`.

Signatures are *not* rebuilt when seeding. Building MinHash signatures costs a
full pass over every document (minutes at 500k docs); exact hashing already
catches the dominant duplication vector, which is the same source being read
twice. Near-duplicate detection still applies among newly streamed documents.

---

## Memory

Near-duplicate removal retains only MinHash signatures, never raw shingle sets.
The defaults are sized for a ~1.3M-document archive on a 16 GB machine:

- `num_perm=16`, `bands=4` (16 perms / 4 rows → ~0.88 recall at Jaccard 0.80,
  threshold `(1/4)^(1/4) ≈ 0.71`).
- Signature storage is ~0.6 KiB/document, so ~1.3M documents ≈ 1.2 GiB in
  signatures, plus ~0.2 GiB of exact hashes.

The earlier `num_perm=32, bands=8` settings cost roughly double and were the
reason an extension run was OOM-killed at ~1.3M documents. If you raise the
archive size substantially, either lower `num_perm` or cap the process:

```bash
systemd-run --user --scope -p MemoryMax=3G -p MemorySwapMax=1G \
    python scripts/expand_tamil_corpus.py --out-dir data/raw/expanded --fineweb-limit 300000
```

Shards are appended and flushed per document, so an interrupted run leaves a
valid (short) trailing shard rather than a truncated line — re-running resumes
cleanly thanks to seeding.

---

## Output

```
data/raw/expanded/
├── corpus-00000.jsonl … corpus-000NN.jsonl   # append-only shards
├── state.json                                # done sources, kept, shards
└── stats.json                                # datasheet for the whole archive
```

Each record:

```json
{"text": "...", "url": "...", "source": "fineweb-2-ta", "tier": 3,
 "quality_score": 0.95, "id": "fineweb-2-ta-157979", "chars": 9567}
```

`stats.json` is **recomputed from every shard** at the end of each run, so it
always describes the full archive (`documents`, `characters`, `shards`,
`source_counts`, `tier_counts`) rather than only the current run's additions.
`dropped` is the exception: it is per-run.

Token estimates use a measured **0.311 swaram tokens/char** (800-document sample
with the frozen tokenizer). The previous hard-coded `0.95` assumed ~1 token per
character and over-reported corpus size roughly 3×.

---

## Reference run

A full expansion produced the following archive (kept locally under `data/`,
which is gitignored):

| Metric | Value |
|---|---|
| Documents | 1,334,070 |
| Characters | 1,055,789,601 |
| Est. swaram tokens | ~328.4M |
| Shards | 71 |
| On-disk size | 2.9 GB |

Source mix: IndicCorp 971,911 · FineWeb-2 157,703 · Wikipedia 123,678 · local
Sangam corpus ~80k. Tier mix: tier 3 = 1,129,614, tier 2 = 123,678, tier 1 =
80,053.

The FineWeb-2 pass in that run stopped early at ~157k of a 400k target when the
OOM killer took the process; the archive itself was intact and `fineweb2` was
left unmarked in `state.json`, so the pass can be resumed.

The archive is **append-only**: the nightly `scripts/cron_yazhi_one_ingest.sh` run
adds whatever is new from the local roots (it added 35 documents on 2026-10-24
without touching the 1.33M already seeded). Quote the numbers with a date, or read
`stats.json`, rather than treating this table as a constant.

### Nightly pipeline gotcha (observed, fixed)

The cron job's packing step used `prepare_slm_corpus.py`, which materialises every
document in RAM *and* runs a `num_perm=128` near-duplicate pass over them. Against
this archive that reached **9.1 GiB** and was still going while a training run was
live — an OOM waiting to happen, on a 16 GB machine. The job now bounds itself with
`--limit` (`ADHAN_PACK_DOCS`, default 50k) and warns when asked to do the whole
archive. The unbounded path for real packing is
`scripts/freeze_fast_tokenizer.py`, which streams and writes in chunks.

---

## Migration from the legacy pipeline (2026-10)

Two earlier scripts were removed because their dedup was O(n²) and they are
fully superseded by this one:

| Removed | Superseded by |
|---|---|
| `scripts/ingest_all_prioritized_data.py` | `expand_tamil_corpus.py` (streaming, LSH) |
| `scripts/phase2_corpus_build.py` | `expand_tamil_corpus.py` |
| `scripts/prepare_full_dataset.py` | dead one-off, no remaining callers |
| `scripts/merge_corpora.py` | byte-identical duplicate of `src/data_scraper/merge_corpora.py` |

Callers were repointed: `scripts/cron_yazhi_one_ingest.sh` now runs
`expand_tamil_corpus.py` → `phase2_validate.py` → `prepare_slm_corpus.py`
against `data/raw/expanded`, and `adhan mutate dedup` runs `scripts/run_dedup.py`
(the unit-tested dedup path).

`phase2_validate.py` now accepts a **directory of shards** plus `--max-records`,
because it holds the corpus in memory and cannot load a 1.3M-document archive
whole.

Derived raw dumps that duplicated the archive were deleted after verifying
coverage: `data/raw/unified_prioritized` (96.5% of its quality-passing documents
already present), `data/raw/thirukkural` (100%), and the empty
`data/raw/{social,reddit,twitter}` trees. The 690 documents that were *not* yet
in the archive (open-sangam classical texts) were appended as a shard before
deletion, so nothing was lost.

Known regenerable assets that intentionally stay under `data/`:

- `data/raw/thirukkural/thirukkural_harness_data.json` — eval harness pairs, used
  by `adhan_slm.eval.thirukkural_eval`. Regenerate with
  `python src/data_scraper/raw_extractors/thirukkural_extractor.py`; its sibling
  `thirukkural_corpus.jsonl` is not kept because it is already in the archive.
- `scripts/benchmark_tokenizers.py` re-fetches its frozen Thirukkural corpus if
  the cache is absent.

---

**Next in the stack:** the tokenizer that consumes this archive — [`TOKENIZER_BENCHMARK_V2.md`](../tokenizer/TOKENIZER_BENCHMARK_V2.md) (Rust akshara BPE, byte fallback, morphology layer), then [`CONTINUED_PRETRAIN.md`](../training/CONTINUED_PRETRAIN.md) for pretraining.

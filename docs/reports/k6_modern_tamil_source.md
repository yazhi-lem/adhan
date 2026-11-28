# K6 — Modern Tamil eval source: options and recommendation

**Date:** 2026-11-18
**Task:** pick, licence-log and fetch the `modern` register of the frozen tokenizer benchmark
([#112](https://github.com/yazhi-lem/adhan/issues/112), K6). The `classical` and `tanglish`
sources are already chosen; the modern one was the last open slot in
`eval_manifests/tokenizer_bench.sources.json`.
**Status:** source chosen and measured here; fetch + freeze implemented in
`scripts/fetch_modern_tamil.py` and `eval_manifests/tokenizer_bench.sources.json`.

---

## 1. What the benchmark needs

`docs/EVAL_CORPUS.md` fixes the register definition: contemporary Tamil prose, at least 80%
Tamil-script letters, 15–300 characters per sentence, 500 sentences chosen deterministically
by `sha256(seed | source | text)`. Because the sample is only 500 sentences from a large
corpus, the requirements that actually bind are **licence clarity, a pinnable revision and a
clean text pipeline** — not raw volume.

Everything committed is counts, hashes and provenance. The raw text stays in git-ignored
`data/`, so no share-alike or attribution obligation is triggered by this repo; the manifest
records what the source is and under which licence.

## 2. Options considered

| Option | What it is | Licence | Size to fetch | Verdict |
|---|---|---|---|---|
| **`wikimedia/wikipedia`, config `20231101.ta`** (HuggingFace mirror of the Wikimedia dump) | Full Tamil Wikipedia article text, snapshot-dated, columns `id/url/title/text` | **CC BY-SA 4.0 (since June 2023) + CC BY-SA 3.0 (older) + GFDL** | **265 MB** (2 parquet) | **Recommended.** Encyclopaedic modern Tamil prose, immutable snapshot, smallest fetch that still gives huge headroom |
| `ai4bharat/sangraha`, `verified/tam` | AI4Bharat's curated verify pass over Indic web text (news, blogs, wiki) | **CC-BY-4.0** (no share-alike) | **18.71 GB** (53 parquet) | Best licence and register diversity, but 70× the bytes for the same 500 sentences. **Documented fallback** if reviewers want web/news register and a non-SA licence |
| Live MediaWiki API via `src/data_scraper/raw_extractors/wikipedia_api_extractor.py` | Fetches articles at request time | Same as Wikipedia | tiny | **Reject.** Not reproducible: the article set changes between runs, so `--verify` can never pass |
| Tamil news sites / RSS | Contemporary prose, closest to "what people read" | Site-specific, mostly all-rights-reserved | small | **Reject.** No reusable licence |
| Project Madurai | Public-domain Tamil literary texts | Public domain | small | **Reject for this register.** Literary/older prose, overlaps `classical`, not modern usage |
| `oscar-corpus/OSCAR-2301` (ta) | Web crawl | Aggregated "other" terms, unclear upstream | GBs | **Reject until terms are explicit** (same failure mode as the `Tngarg` Tanglish candidate) |

## 3. Chosen source

```text
repo:      wikimedia/wikipedia
config:    20231101.ta
revision:  b04c8d1ceb2f5cd4588862100d08de323dccfbaa
files:     20231101.ta/train-00000-of-00002.parquet   146.4 MB
           20231101.ta/train-00001-of-00002.parquet   119.2 MB
licence:   CC BY-SA 4.0 (edits since June 2023) and CC BY-SA 3.0 (older revisions);
           all also GFDL (Wikimedia dump terms)
url:       https://huggingface.co/datasets/wikimedia/wikipedia
```

The snapshot config name carries the dump date, so the dataset is immutable in practice and
the file is additionally pinned by revision. Only the first shard is fetched by default (the
second is available via `--shard 1`); one shard already yields far more than 500 candidates.

### Why this one

- **Fit:** encyclopaedic Tamil prose is the cleanest available *modern* register — no
  comment-section noise, no transliteration, real orthography and punctuation.
- **Reproducibility:** dump-dated snapshot plus a pinned revision means the `sha256` in the
  manifest can be rebuilt by anyone, unlike the live API.
- **Cost:** 146 MB for one shard. Sangraha would be 18.71 GB for the same 500 sentences.
- **Failure mode is loud:** a changed upstream file breaks the manifest hash and `--verify`
  fails, instead of silently changing the sample.

### Why not Sangraha as primary

It is the better *licence* (CC BY 4.0, no share-alike) and a more diverse register. It is not
the better *choice for 500 sentences*: the fetch is 70× larger and the verified subset needs
the same parquet tooling anyway. It stays documented as the fallback if a reviewer objects to
CC BY-SA, or later when a second modern source is wanted for a cross-register A/B.

## 4. Tooling cost

The dump ships as parquet, so the fetch step needs `pyarrow`. It is an **optional extra**,
not a core dependency:

```bash
pip install -e ".[eval-corpus]"
python scripts/fetch_modern_tamil.py          # writes data/raw/modern_tamil/wikipedia_ta.jsonl
python scripts/build_eval_corpus.py           # freeze
python scripts/build_eval_corpus.py --verify
```

This generalises: any HuggingFace parquet text source (Sangraha, `wikimedia/wikipedia` for
other languages, IndicGLUE) can be converted with the same small script, so the eval corpus
is not welded to this one dataset.

## 5. Risks and how to handle them

| Risk | Why it matters | Mitigation |
|---|---|---|
| **CC BY-SA share-alike** | The dump mixes revisions: edits since June 2023 are CC BY-SA 4.0, older ones CC BY-SA 3.0, all also GFDL. A derived *published* sample would carry those obligations | Only counts, hashes and provenance are committed; raw and sampled text stay in git-ignored `data/`. Every record keeps `title`, `page_id` and the dump `snapshot`, so attribution is possible without shipping text |
| **MediaWiki markup leaking into sentences** | Wiki syntax would distort token counts | The dump's `text` field is plain text (no wikitext); the eval-corpus rules still drop tables, references and short fragments by length and script share |
| **Stale snapshot** | Wikipedia moves on | The dump date is in the config name and pinned by revision; a newer dump is a new source, not an edit |
| **Encyclopaedic-only register** | Formal prose may not represent everyday modern Tamil | Documented as a limitation; Sangraha (web/news) is the fallback and can be added as a second modern source later |
| **Parquet tooling** | A new optional dependency | Confined to the fetch script and an opt-in extra; the corpus builder itself stays standard-library only |

## 6. Decision

Adopt **`wikimedia/wikipedia` `20231101.ta` at `b04c8d1`** as the `modern` source, fetched
with `scripts/fetch_modern_tamil.py` and pinned by file SHA-256 in the manifest. Keep
**`ai4bharat/sangraha` `verified/tam` at `8b813c3`** documented as the fallback for a
non-share-alike or web/news register. Do not use the live Wikipedia API, Tamil news sites or
OSCAR.

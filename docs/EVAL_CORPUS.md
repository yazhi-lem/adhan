# Frozen eval corpus for the Tamil tokenizer benchmark

Part 1 of [#74](https://github.com/yazhi-lem/adhan/issues/74). The corpus is the fixed input
for the public tokenizer benchmark. It has three registers:

| Register | What it is |
|---|---|
| `classical` | Sangam / Thirukkural-style Tamil |
| `modern` | contemporary Tamil prose (news, Wikipedia, ...) |
| `tanglish` | Tamil written in Latin script |

## How it is built

`scripts/build_eval_corpus.py` reads a **spec** (`eval_manifests/tokenizer_bench.sources.json`)
that lists every source with its register, path, licence and URL, then:

1. reads each source (`jsonl` with a `text_field`, or `txt` with one document per line)
2. NFC-normalises, collapses whitespace and splits into sentences (`.`, `?`, `!`, `।`, newline)
3. keeps sentences of `min_chars`..`max_chars` in the right script
   (classical and modern: at least 80% Tamil letters; tanglish: no Tamil-script characters
   and at least 80% Latin letters)
4. drops exact duplicates across all sources
5. applies the **content filter** (see below) and drops every sentence it vetoes
6. ranks the remaining sentences by `sha256(seed | source | text)` and keeps the first
   `per_register` of each register

Step 6 makes the sample a pure function of the text and the seed: input order, file order
and platform do not change it. The output is canonical JSONL (`data/eval/tokenizer_bench.jsonl`,
git-ignored) and a manifest.

## Content filter (K6 / #112)

The public set must not ship an inflammatory, political or sexual sentence scraped from
social media. A spec may name a gazetteer:

```json
{ "content_filter": "eval_manifests/content_blocklist.v2.json" }
```

The path resolves like a source path; `"none"` (the default) applies no filter. Matching is
deterministic and deliberately simple (`src/adhan_slm/eval/content_filter.py`):

- Latin **exact** terms match case-insensitively as whole words, so `vote` does not fire
  inside `devotee`; internal whitespace in a term matches any run of whitespace.
- Latin **stems** may also take a closed set of Tanglish case suffixes, which is how Tanglish
  inflects: `pundai` + `ku`, `mayir` + `um`, `arasiyal` + `uku`. The set is closed on purpose —
  an open `\w*` would drag in `warning` for `war`.
- Tamil terms match at a word start and may be followed only by a case/number marker
  (`தேர்தல்` → `தேர்தலில்`, `அமைச்சர்` → `அமைச்சரை`). A term ending in pulli is indexed with and
  without it, because Tamil drops the pulli before a suffix. The trailing boundary is what stops
  `சாதி` firing inside `சாதிக்க` and `புண்ட` inside `புண்டரீகாட்சன்`.
- All terms and all input are NFC-normalised, so decomposed text cannot slip past.
- Patterns are compiled once when the gazetteer loads, not per sentence.
- Categories are checked in file order and the first hit wins, so the same sentence always
  reports the same category.

The gazetteer is JSON rather than code so a native reviewer can extend it without touching
Python: [`../eval_manifests/content_blocklist.v2.json`](../eval_manifests/content_blocklist.v2.json).
v2 is a seed list from the N3 scope report, extended after review to catch inflected forms and to stop matching inside longer Tamil words, and is marked `"native_review": "pending"`; it
must be reviewed and extended before the corpus is frozen for publication.

The manifest records the filter's version, SHA-256, categories and native-review state, plus
how many candidates each source lost per category, e.g.

```json
"content_filter": { "version": "v2", "sha256": "…", "categories": [
  "politics", "violence", "sexual", "abuse"], "native_review": "pending" },
"counts": { "per_source": { "tanglish-social": { "blocked": { "politics": 3401 } } } }
```

`--verify` also fails if the gazetteer changed under a saved manifest, even when the
version string is unchanged.

## The manifest

`eval_manifests/tokenizer_bench.manifest.json` is committed. It records the exact spec, the
SHA-256 of every input file, the licence and URL of every source, per-source counts
(documents, sentences, candidates, selected) and the SHA-256 of the output. Anyone can check
a rebuild:

```bash
python scripts/build_eval_corpus.py --verify
```

`--verify` rebuilds from the raw inputs and fails if an input changed, the counts differ, the
output hash differs, the file on disk differs or any text is not NFC.

## Licences

A source whose `license`, `url` or `path` is empty or starts with `TODO` is rejected, so an
unlicensed source cannot enter the public set. Keep unlicensed text out of the committed
manifest and out of any public repo.

## Sources (pinned)

`eval_manifests/tokenizer_bench.sources.json` is committed and names three sources, one per
register. Every one is pinned so a rebuild matches the manifest hash:

| Register | Source | Pin | Licence |
|---|---|---|---|
| `classical` | Thirukkural corpus (`tk120404/thirukkural`) | local file SHA-256 | Kural verses public domain; repo data Apache-2.0; embedded M. Varadarasanar commentary is third-party and is not relicensed |
| `modern` | `wikimedia/wikipedia`, config `20231101.ta` | revision `b04c8d1` + file SHA-256 | CC-BY-SA-3.0 + GFDL |
| `tanglish` | `vishnu-n/Tanglish-Corpus-185k` | revision `927f5b4` + file SHA-256 | CC-BY-4.0 |

Why these three: [`reports/k6_modern_tamil_source.md`](reports/k6_modern_tamil_source.md) and
[`reports/n3_tanglish_source_scope.md`](reports/n3_tanglish_source_scope.md).

## Using it

```bash
pip install -e ".[eval-corpus]"        # pyarrow, only for the dump-backed fetch
python scripts/fetch_modern_tamil.py   # data/raw/modern_tamil/wikipedia_ta.jsonl
python scripts/build_eval_corpus.py    # freeze + write the manifest
python scripts/build_eval_corpus.py --verify
```

The classical and Tanglish raw files are fetched separately (Thirukkural extractor, pinned HF
resolve URL). `data/` is git-ignored, so a clean clone must fetch the three raw inputs before
`--verify` can pass; the manifest records the SHA-256 each one must have.

If a register has fewer usable sentences than `per_register`, the build fails. Add sources,
lower `per_register`, or pass `--allow-shortfall`.

### Current freeze

`eval_manifests/tokenizer_bench.manifest.json` was built from the sources above: 1,500
sentences (500 per register), output SHA-256 `003a2dfe…`. Candidates that survived the rules:
classical 8,107 · modern 1,406,763 · tanglish 238,817. The content filter vetoed 145
(classical), 82,532 (modern) and 13,616 (tanglish) candidates, so the 500-sentence sample is
not drawn from political or abusive sentences in any register.

## Data handling

The builder is pure Python (standard library), does no network I/O and sends no text
anywhere. Fetching raw sources is a separate step (`scripts/fetch_modern_tamil.py` for the
modern register, `src/data_scraper/raw_extractors` for the rest). The tests use small
synthetic fixtures only.

## Still needed (not done by code)

- A native reviewer signs off the v2 gazetteer (`eval_manifests/content_blocklist.v2.json`)
  and reads the 500 sampled Tanglish sentences before the corpus goes public.
- Provenance review of the Tanglish source (see the N3 report §5): only the DravidianCodeMix
  subset is CC-BY-4.0; the rest are YouTube and Reddit comments under platform terms.
- Re-run `--verify` after any gazetteer change; the manifest records the filter hash, so a new
  gazetteer version is a new freeze, not an edit.

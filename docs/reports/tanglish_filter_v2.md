# Content filter v2: Tanglish source measurement

Task: [K6 / #112](https://github.com/yazhi-lem/adhan/issues/112). Answers one
question with committed code: after the eval-corpus rules and the gazetteer pass,
how many neutral Tanglish sentences remain to sample 500 from?

## Source (pinned)

| Field | Value |
|---|---|
| Path | `data/raw/tanglish/tanglish_corpus.jsonl` (git-ignored, local only) |
| SHA-256 | `cd1f7d589cf89f6f59be81099dfda928f9e9dd0e47b6c95a3ba0d7df7f2b02bd` |
| Size | 27,854,538 bytes |
| Repo | `vishnu-n/Tanglish-Corpus-185k` @ `927f5b4` |
| Declared licence | CC-BY-4.0 (dataset card) |

## Rules applied

- 15–300 characters per sentence
- `tanglish` script rule: no Tamil-script characters, at least 80% Latin letters
- exact-duplicate removal across the source
- content filter `v2` (categories: politics, violence, sexual, abuse)

## Results

| Stage | Count |
|---|---:|
| Raw rows | 185,973 |
| Sentences | 267,040 |
| Dropped: length | 13,415 |
| Dropped: script | 5 |
| Dropped: duplicate | 1,190 |
| Dropped: content filter | 12,776 |
| **Neutral candidates** | **239,654** |

### Content filter, per category

| Category | Blocked |
|---|---:|
| politics | 9,145 |
| violence | 1,198 |
| sexual | 359 |
| abuse | 2,074 |

The filter vetoed **5.1%** of candidates. That matches the
~5.5% keyword-positive rate in the N3 scope report
([`n3_tanglish_source_scope.md`](n3_tanglish_source_scope.md)), so the filter behaves as
scoped rather than over- or under-filtering.

At 239,654 neutral candidates the 500-sentence sample has
**479× headroom**, so the freeze does not depend on a handful of
rows and can survive a stricter reviewer-passed gazetteer.

## Caveats

- `native_review` is `pending`: this is a seed gazetteer.
  A native reviewer must extend and sign it off before the corpus is frozen.
- A gazetteer is high recall, not precision. It removes some innocent rows and misses
  paraphrases; MuRIL was probed for this and rejected as too weak to ship
  ([`muril_base_assessment.md`](muril_base_assessment.md)).
- Counts describe this pinned revision only; a newer upstream file is a different
  corpus and needs its own measurement.

Regenerate: `python scripts/tanglish_filter_report.py`

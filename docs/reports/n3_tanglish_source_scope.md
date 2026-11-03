# N3 — Tanglish eval source: options and recommendation

**Date:** 2026-11-03
**Task:** unblock the `tanglish` register of the frozen tokenizer benchmark (`factory/NEXT.md` N3)
**Status:** scope complete; selection proposed, implementation not started

---

## 1. What the benchmark needs

`docs/EVAL_CORPUS.md` requires one licensed source for each of three registers. The
`tanglish` register is defined narrowly:

- Latin script only: **zero Tamil Unicode characters**;
- at least 80% Latin letters;
- 15–300 characters per sentence;
- 500 selected sentences, deterministically ranked by `sha256(seed | source | text)`;
- source path, **licence**, and URL recorded in the manifest;
- raw text stays out of the repo (`/data/` is git-ignored); only the manifest is committed.

The current blocker is not code — it is a source that is both **real Tanglish** and
**licence-clean enough to cite publicly**.

## 2. Options considered

| Option | What it is | Licence | Fit | Verdict |
|---|---|---|---|---|
| **`vishnu-n/Tanglish-Corpus-185k`** (HuggingFace) | 185,973 real Romanised Tamil–English sentences scraped from YouTube, Reddit and DravidianCodeMix; JSONL with `text` + `source` | **CC-BY-4.0** (dataset card) | **Strong.** Measured 252,405 candidate sentences after the eval corpus filters — 500× headroom. Real code-mixing, Latin-only, sentence-level | **Recommended primary** |
| **DravidianCodeMix / DravidianLangTech (FIRE 2020+)** | Official shared-task Tamil–English code-mixed corpora, ~15.7k sentences | Research-use terms; not a blanket redistribution licence | Good linguistic fit, but smaller and gated | **Fallback if the primary fails provenance review** |
| **`Tngarg/Codemix_tamil_english`** (HuggingFace) | 1.8 MB CSV of Tamil–English code-mixed rows | `license: other` with no readable terms | Unknown licence | **Reject until terms are explicit** |
| **`Deepakvictor/tanglish-tamil`** (HuggingFace) | 597 rows of song lyrics from karky.in with Tamil/Tanglish pairs | OpenRAIL tag, but lyrics are third-party copyrighted text | Wrong register (lyrics), small, provenance risk | **Reject** |
| **`goru001/nlp-for-tanglish`** (GitHub) | MIT code plus links to DravidianCodeMix and transliterated Wikipedia | MIT for code; underlying data inherited | Does not ship a standalone corpus | **Useful pointer, not a source** |
| **Synthetic romanisation** from a permissive Tamil corpus using the Azhagi/Open-Tamil scheme (MIT) | Deterministic Tamil → Latin transliteration | Fully controlled (MIT + source corpus) | Reproducible and licence-clean, but **not natural Tanglish**; no English code-mixing | **Last-resort fallback**, label as `romanized_tamil`, not `tanglish` |
| Unlicensed scrapes / Alpaca-style synthetic Tanglish | Various | None / unclear | Not auditable | **Reject** |

## 3. Recommended source

**`vishnu-n/Tanglish-Corpus-185k`**, pinned to revision:

```text
repo:      vishnu-n/Tanglish-Corpus-185k
revision:  927f5b497b9f9e35ffc2a10b621a97f6e02ac9c2
file:      tanglish_corpus.jsonl
size:      27,854,538 bytes
rows:      185,973
licence:   CC-BY-4.0 (as declared by the dataset card)
url:       https://huggingface.co/datasets/vishnu-n/Tanglish-Corpus-185k
```

### Measured viability

I downloaded the pinned JSONL to `/tmp` (not committed) and applied the eval-corpus rules:

| Measure | Value |
|---|---|
| Raw rows | 185,973 |
| Sentences after split + 15–300 chars + script check | 252,405 |
| Distinct sentences after exact dedupe | 252,405 |
| Needed for the benchmark | 500 |
| Headroom | ~505× |

Sample rows are genuine Tanglish:

```text
"bro office ku late ah vanthen romba tired ah iruku"
"machan salary cut pannanga romba tension ah iruku"
"GPUku heat adhigam iruku cooling fix pannanum"
```

### Why it fits

- It is **real** social-media Tanglish, not machine-translated or synthetic.
- It is **sentence-level** and Latin-only, exactly the register the benchmark defines.
- The corpus card and HF metadata agree on **CC-BY-4.0**.
- Its size means the 500-sentence sample is stable even after a content filter.

## 4. Risks and how to handle them

| Risk | Why it matters | Mitigation |
|---|---|---|
| **Provenance of scraped comments** | The uploader declares CC-BY-4.0, but YouTube/Reddit comments are user-generated; the uploader may not hold the right to relicense them | Treat the raw file as **local-only**. Publish the manifest (URL + pinned revision + SHA-256 + declared licence), not the text. Get a legal/provenance review before the benchmark goes public. Keep the fallback ready. |
| **Political, abusive or profane content** | The raw sample includes Tamil Nadu political discussion and profanity. A public tokenizer benchmark should not ship a random inflammatory sentence. | Add a deterministic **exclusion blocklist** (politics, abuse, sexual content, slurs) to the spec/builder; record the filter version in the manifest. Re-check the selected 500. |
| **Domain tilt** | Most rows come from YouTube channels and one subreddit; register may be narrower than “Tanglish” generally | Balance by `source` where possible; report the source histogram in the manifest. Do not claim population-level coverage. |
| **Upstream drift or deletion** | A moving HF dataset breaks `--repro` | Pin the revision above; record the file SHA-256; keep a local copy outside git for reproduction. |
| **Licence mis-tagging** | A dataset card is not a legal opinion | Document the declared licence verbatim; do not relicense the text; attribute the dataset in the results table and manifest. |

## 5. Implementation plan (when N3 proceeds)

1. **Fetch and pin**
   ```bash
   mkdir -p data/raw/tanglish
   curl -L --fail \
     'https://huggingface.co/datasets/vishnu-n/Tanglish-Corpus-185k/resolve/927f5b497b9f9e35ffc2a10b621a97f6e02ac9c2/tanglish_corpus.jsonl' \
     -o data/raw/tanglish/tanglish_corpus.jsonl
   sha256sum data/raw/tanglish/tanglish_corpus.jsonl
   ```
2. **Add the source to `eval_manifests/tokenizer_bench.sources.json`**
   ```json
   {
     "id": "tanglish-social",
     "register": "tanglish",
     "path": "data/raw/tanglish/tanglish_corpus.jsonl",
     "format": "jsonl",
     "text_field": "text",
     "license": "CC-BY-4.0 (dataset card, vishnu-n/Tanglish-Corpus-185k)",
     "url": "https://huggingface.co/datasets/vishnu-n/Tanglish-Corpus-185k"
   }
   ```
3. **Add a deterministic content filter** to `src/adhan_slm/eval/eval_corpus.py`
   (spec field `exclude_patterns` or a named `content_filter`), so the 500 are reproducible
   and inflammatory rows cannot slip in. Record the filter name/version in the manifest.
4. **Build and freeze**
   ```bash
   python scripts/build_eval_corpus.py
   python scripts/build_eval_corpus.py --verify
   ```
5. **Native review** of the 500 selected Tanglish sentences before committing the manifest.
6. **Publish attribution** next to the canonical tokenizer table:
   `Tanglish sample: vishnu-n/Tanglish-Corpus-185k, CC-BY-4.0, revision 927f5b4`.

## 6. Decision

Adopt **`vishnu-n/Tanglish-Corpus-185k` at revision `927f5b4` as the primary Tanglish
source**, with a deterministic content filter and a local-only raw copy. If provenance
review rejects the source, fall back to the official DravidianCodeMix corpora. Do not use
`Tngarg` (no readable terms), song lyrics, or synthetic Tanglish for the public table.

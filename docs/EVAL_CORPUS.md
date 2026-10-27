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
5. ranks the remaining sentences by `sha256(seed | source | text)` and keeps the first
   `per_register` of each register

Step 5 makes the sample a pure function of the text and the seed: input order, file order
and platform do not change it. The output is canonical JSONL (`data/eval/tokenizer_bench.jsonl`,
git-ignored) and a manifest.

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

## Using it

```bash
cp eval_manifests/tokenizer_bench.sources.example.json eval_manifests/tokenizer_bench.sources.json
# fill in the real sources, licences and URLs, then:
python scripts/build_eval_corpus.py
python scripts/build_eval_corpus.py --verify
```

If a register has fewer usable sentences than `per_register`, the build fails. Add sources,
lower `per_register`, or pass `--allow-shortfall`.

## Data handling

The builder is pure Python (standard library), does no network I/O and sends no text
anywhere. Fetching raw sources is a separate step (`src/data_scraper/raw_extractors`). The
tests use small synthetic fixtures only.

## Still needed (not done by code)

- Choose and license-check the modern and Tanglish sources.
- A native reviewer checks the Tamil and Tanglish samples before the corpus is frozen.
- Commit the filled-in spec and the generated manifest.

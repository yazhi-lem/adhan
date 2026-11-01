# Reproduce the Tamil tokenizer benchmark

Part 3 of [#74](https://github.com/yazhi-lem/adhan/issues/74). One command rebuilds the
frozen eval corpus, runs every tokenizer and writes (or checks) the results tables.

```bash
scripts/reproduce_benchmark.sh            # rebuild everything and write the tables
scripts/reproduce_benchmark.sh --check    # rebuild and compare with the committed tables
```

The same thing without the shell wrapper: `PYTHONPATH=src python scripts/reproduce_benchmark.py`.

## What it does

1. Checks the pinned library versions in the config (`pins`).
2. Fetches any raw source that is missing from `data/` (only sources with a `fetch` entry,
   for example the Thirukkural dataset). Other missing sources stop the run with a message
   saying where to put the file.
3. Rebuilds the frozen eval corpus ([EVAL_CORPUS.md](EVAL_CORPUS.md)) and compares it with the
   committed manifest.
4. Runs the harness ([TOKENIZER_BENCH.md](TOKENIZER_BENCH.md)) on every tokenizer in the config.
5. Writes the tables, or in `--check` mode compares them with the committed ones and prints
   every difference.

## Config

`eval_manifests/tokenizer_bench.reproduce.json` (copy the `.example.json` and fill it in):

| Key | Meaning |
|---|---|
| `spec`, `manifest`, `corpus` | part 1 spec, committed manifest, generated corpus (git-ignored) |
| `results_json`, `results_md` | the committed result tables |
| `tokenizers` | list of `swaram` (vocab, merges), `tokenizers-json` (path), `transformers-local` (path), `baselines` |
| `pins` | `{"package": "version"}`; a mismatch fails the run |
| `fetch` | `{"source id": "fetcher name"}` for sources that can be downloaded |
| `price_per_million` | optional USD per 1M tokens, adds the cost column |

Tokenizer paths are relative to the repo root. Gemma and Llama tokenizers are excluded.

## Why the tables are reproducible

- The corpus sample is a pure function of the text and a seed (part 1).
- The result files contain no timestamps or timings; floats are rounded to 8 digits.
- If the rebuilt corpus differs from the committed manifest (an input file or the spec
  changed) the run fails. Use `--refreeze` only when the change is intended; then commit the
  new manifest and tables together.
- `--check` fails on any difference in numbers or in the Markdown table.
- `environment` (Python and library versions) is stored in the JSON for information and is
  not compared.

## Flags

| Flag | Effect |
|---|---|
| `--check` | compare with committed tables, write nothing |
| `--refreeze` | accept a changed corpus and rewrite the manifest |
| `--allow-version-drift` | warn instead of failing when pinned versions differ |

## Clean-checkout check

```bash
git clone https://github.com/yazhi-lem/adhan && cd adhan
pip install -e ".[benchmark]"
scripts/reproduce_benchmark.sh --check
```

## Still needed

- Fill in the real modern and Tanglish sources and the Swaram vocabulary paths.
- Commit the generated manifest and result tables, then run the clean-checkout check above.
- Tiktoken support in the harness (no character offsets yet).

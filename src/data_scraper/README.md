# data_scraper — Tamil corpus collection

A collection of scrapers and helpers used to collect, clean and export Tamil text for training the AADHAN model.

This directory contains high‑level site crawlers, PDF extractors and small utilities to produce HuggingFace‑style JSONL datasets and per‑work manifests (Project Madurai / Sangam literature focus).

---

## Minimal Scraper (Recommended)

For quick, high-quality Tamil data collection:

```bash
python src/data_scraper/minimal_scraper.py --domains govt,news,wiki --limit 100
```

Scrapes 300–500 records in <5 minutes from:
- TN Government schemes (official Tamil, quality 0.9)
- Tamil news headlines (modern conversational, quality 0.7)
- Wikipedia current events (formal contemporary, quality 0.8)

Full options:

```bash
python src/data_scraper/minimal_scraper.py \
  --domains govt,news,wiki \
  --limit 100 \
  --output data/scraped/minimal.jsonl
```

Output is JSONL with fields: `text`, `source`, `url`, `quality_score`, `id`.

---

## Contents 📂

- `tamil_corpus_scraper.py` — main crawler and extractor (sitemaps, literature sites, news, social simulacra). Supports FineWeb‑style filtering, HF export, PDF extraction and manifest building.
- `pdf_book_scraper.py` — PDF-focused extractor that downloads/extracts PDFs from sitemaps, converts legacy encodings (TSCII/Mylai) and optionally attempts Tamil→English translation.
- `pmworks_extractor.py` — fast, deterministic extractor for Project Madurai's `pmworks.html` table; builds per‑work JSONL + combined text files.
- `merge_hf_datasets.py` — utility to merge multiple HF JSONL exports, dedupe and re‑export combined train/val/test splits.

---

## Quick start — prerequisites & environment ⚙️

1. Use the project virtualenv (repo contains `.venv`).
   - Activate: `source .venv/bin/activate`
2. Optional language tools (the scraper can auto-install some):
   - `pdfminer.six` (PDF extraction), `tamil` (TSCII conversion), `googletrans` (optional translation).
   - You can run: `python src/data_scraper/tamil_corpus_scraper.py --check_packages` to attempt installation of common packages.

---

## How to run (examples) ▶️

- Basic literature sitemap crawl (Project Madurai, TamilVu, Sangam.org) with FineWeb‑style filtering and HF export:

  ```bash
  python src/data_scraper/tamil_corpus_scraper.py \
    --basic --basic-limit 200 --dedupe --classify --extract-pdfs \
    --build-manifest --to-hf --format jsonl \
    --output data/raw/basic_crawl/combined_basic.jsonl \
    --min-quality 0.5 --min-chars 100 --require-tamil --apply-fineweb
  ```

- Full scraping pipeline (Wikipedia, news, literature + HF export):

  ```bash
  python src/data_scraper/tamil_corpus_scraper.py --full --to-hf --dedupe --classify \
    --min-quality 0.5 --min-chars 100 --require-tamil --apply-fineweb
  ```

- Extract PDFs from specified domains, convert legacy encodings and build manifests:

  ```bash
  python src/data_scraper/pdf_book_scraper.py --domains projectmadurai.org,tamilvu.org --limit 100 --translate
  ```

- Build Project Madurai manifests from previously-saved `pmworks.html`:

  ```bash
  python src/data_scraper/pmworks_extractor.py --input data/raw/raw_html/projectmadurai.org/pmworks.html
  ```

- Merge HF JSONL exports into a single deduped HF dataset:

  ```bash
  python src/data_scraper/merge_hf_datasets.py
  ```

---

## Important CLI options (`tamil_corpus_scraper.py`) 🔧

- `--basic` : one‑shot sitemap crawl for literature sites (Project Madurai, TamilVu, Sangam.org).
- `--basic-limit` : per‑site page/PDF limit (0 = no limit).
- `--extract-pdfs` : download & extract PDF text (requires `pdfminer.six`).
- `--to-hf` : export to HuggingFace‑style JSONL splits (train/validation/test).
- `--dedupe` / `--classify` : run deduplication and keyword topic classification + quality scoring.
- FineWeb‑style filters:
  - `--min-quality <float>` (0.0–1.0) — drop records with low `quality_score`.
  - `--min-chars <int>` — minimum character length for content to keep.
  - `--require-tamil` — require Tamil‑dominant text (filters pages with low Tamil fraction).
  - `--apply-fineweb` — run extra heuristics (sentence count, repetition, punctuation ratios).

Recommended starting filters for training-quality data: `--min-quality 0.5 --min-chars 100 --require-tamil`.

---

## Outputs & where to look 🔎

- Raw HTML saved under: `data/raw/raw_html/<domain>/...` (when `--save-raw-html` used or sitemap crawl saves raw pages).
- Project Madurai manifests: `data/raw/projectmadurai_manifests/` (`manifest_index.json`, per-work `.jsonl` and `.txt`).
- PDF book manifests: `data/raw/pdf_books_manifests/` (per-domain `.jsonl` + per-work files).
- Basic crawl outputs: `data/raw/basic_crawl/` (per-site JSONL + `hf/` when `--to-hf`).
- Final HF dataset(s): `data/raw/combined_hf/` (train/validation/test JSONL).

---

## Notes on Sangam literature, poems & poets 📜✨

Why Sangam matters
- Sangam literature is classical Tamil poetry (anthologies dated roughly from early centuries BCE to early centuries CE). It is a rich source of canonical Tamil text and a valuable part of the training corpus.

Typical sources in this project
- `projectmadurai.org` — large collection of Tamil e-texts (many Sangam works).
- `tamilvu.org` — Tamil Virtual Academy educational resources and classical literature.
- `sangam.org` — focused Sangam resources and commentary.

Data shape and metadata
- `pmworks_extractor.py` parses the Project Madurai `pmworks.html` table and preserves `work_no`, `title`, `author`, `genre`, `pdf_links`, and `html_links`.
- Per‑work manifests created by the scraper include `work_id`, `n_pages`, `char_count`, and `records` that keep `author`/`title` when available.
- Poems are often short (couplets or short stanzas). We preserve line breaks in per‑work `.txt` files and record-level `text` fields.

Handling poets & attribution
- `author` metadata (when present) is preserved in manifests and per-record JSONL.
- For poems without explicit author (common in ancient anthologies), `author` may be empty or listed as the anthology/collection. Use `work_id` + `title` to group/identify poems.

Special handling for poems
- Keep short-form texts (couplets) — do **not** force aggressive sentence-splitting on Sangam poetry.
- Use `--min-chars` conservatively if you want to include short poems (e.g. set `--min-chars 20` to retain short verses).
- TSCII / legacy encodings: older e-texts may use legacy encodings — `pdf_book_scraper.py` and `pmworks_extractor.py` include heuristics to detect & convert TSCII.

---

## Quality & filtering guidance (FineWeb‑style) ✅

- Use `--require-tamil` to ensure the text is Tamil‑dominant.
- `--min-quality` filters by the scraper's `score_quality()` heuristic (length + Tamil fraction). 0.5 is a reasonable starting point.
- `--apply-fineweb` enables stricter heuristics (sentence count, repetition checks) to reduce boilerplate/noise.
- Always run `--dedupe` after large crawls to remove exact & near duplicates.

---

## Troubleshooting & tips ⚠️

- PDF extraction fails → install `pdfminer.six`: `pip install pdfminer.six`.
- Legacy encoding issues → ensure `tamil` package is installed (`pip install tamil`) or run `pdf_book_scraper.py` which attempts detection/conversion.
- Robots / rate limits → the crawler respects `robots.txt`; use `--basic-limit` to reduce load and avoid blocking.
- Inspect raw HTML for debugging: use `--save-raw-html` to keep downloaded pages under `data/raw/raw_html/`.

---

## Next steps / suggestions ✨

- Manually review a sample of per‑work `.txt` files (in `data/raw/projectmadurai_manifests/`) before model training to ensure poetry formatting is preserved.
- After HF export, run a tokenizer pass and create training shards for your training pipeline.
- Consider building a small human‑verified validation set of Sangam poems for evaluation.

---

## License & credits

- Data sources are public‑domain / site‑specific — verify each source's license before redistribution.
- The scraping code and heuristics are authored in this repository.

---

If you want, I can:
- add a short example script to sample + inspect a few Sangam works, or
- add a small `requirements-dev.txt` listing recommended packages.

Choose one and I will add it now.
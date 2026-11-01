# data_scraper — Tamil corpus collection (legacy + Phase 2 tools)

Scrapers, importers and exporters used to collect Tamil text. **The production corpus is now built by
`scripts/expand_tamil_corpus.py` / `expand_tamil_corpus_v2.py` and the library crawlers**
(`docs/corpus/CORPUS_EXPANSION.md`, `docs/corpus/DIGITAL_LIBRARIES.md`). Use this package for targeted collection and imports.

## Layout

| Path | What |
|---|---|
| `minimal_scraper.py` | Quick collection: TN government schemes, news headlines, Wikipedia current events → JSONL (`text, source, url, quality_score, id`) |
| `raw_extractors/` | Site crawlers: `tamil_corpus_scraper.py` (sitemaps, literature sites, FineWeb-style filtering, HF export), `pdf_book_scraper.py` (PDF + TSCII/Mylai conversion), `pmworks_extractor.py` (Project Madurai `pmworks.html`), `thirukkural_extractor.py`, `wikipedia_api_extractor.py`, `reddit_scraper.py`, `twitter_scraper.py` |
| `importers/` | Sibling corpora: `sangam_importer.py` (Open Sangam), `vazhi_importer.py`, `corpus_tamil_importer.py` |
| `processing/` | `build_unified_corpus.py`, `build_pretraining_sentences.py`, `extract_wiki_sentences.py`, `merge_hf_datasets.py` |
| `export/` | `export_unified_hf.py` (HF-style JSONL splits) |
| `evaluation/` | `validate_tamil_corpus.py`, `compare_corpus.py`, `analyze_tamil_style.py` |
| `merge_corpora.py` | ADHAN + VAZHI merge (see `docs/DEV.md` §6) |

## Run

```bash
python src/data_scraper/minimal_scraper.py --domains govt,news,wiki --limit 100 --output data/scraped/minimal.jsonl
python src/data_scraper/raw_extractors/tamil_corpus_scraper.py --basic --basic-limit 200 --dedupe --classify --extract-pdfs
python src/data_scraper/raw_extractors/pmworks_extractor.py --input data/raw/raw_html/projectmadurai.org/pmworks.html
python scripts/run_scraper.py --social reddit --social-max-posts 500        # social collection (Phase 2)
```

Every module has `--help`. Optional tools: `pdfminer.six` (PDFs) and `tamil` (TSCII conversion).

## Rules

- **Quality:** keep `--min-quality` at about 0.5 (length + Tamil fraction); `--apply-fineweb` adds sentence and
  repetition checks. Tamil-script share ≥ 0.80 for the training corpus.
- **Sangam:** keep `work_no`, `title`, `author`, `genre` from Project Madurai manifests. For Open Sangam, ingest one
  document per (song, layer) only (`docs/corpus/SANGAM_INGEST.md`), never merged blocks.
- **Be polite and legal:** respect robots.txt, throttle, and honour each site's licence. Record the source + licence in
  every record and datasheet.

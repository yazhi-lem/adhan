# Digital Library Ingestion — Tamil corpus sources beyond web crawl

Adhan's corpus is built from HF datasets (IndicCorp, FineWeb-2, mC4, Wikipedia)
plus **digitised Tamil libraries**. This document covers the library crawlers,
what they reach, and how to run them.

## Sources

| Source | Script | What it provides | Licence / status |
|---|---|---|---|
| Tamil Virtual Academy (`tamilvu.org`) | `scripts/ingest_tamilvu.py` | Sangam works with commentary, grammar texts, epics, dictionaries, encyclopedias, modern literature, folklore | Government of Tamil Nadu; HTML pages |
| Tamil Wikisource (`ta.wikisource.org`) | `scripts/ingest_wikisource_ta.py` | Public-domain Tamil texts via the MediaWiki API | CC-BY-SA |
| Project Madurai | `scripts/ingest_tamil_web.py` | Classical and modern Tamil literature | Public domain / CC |
| HF datasets | `scripts/expand_tamil_corpus_v2.py` | IndicCorp v2, FineWeb-2, mC4, Wikipedia | dataset-specific |

## Tamil Virtual Academy detail

The catalogue is `/ta/library-libcontnt-273141`; it exposed **352 works**. Each
work is a frameset:

* left frame `/slet/<code>/<code>lft.jsp` — table of contents
* chapter pages `<code>son.jsp?subid=N` — verses
* commentary popups `<code>uri.jsp?...` — behind an author-selection flow
  (`uri.jsp -> sec.jsp`), enabled with `--with-commentary`

The crawler resolves the TOC, follows chapter pages, strips site chrome, and
writes Tamil prose. It is resumable via `--state` and polite via `--delay`.

```bash
# high-value works (Sangam, Kural, epics, grammar, dictionaries)
python scripts/ingest_tamilvu.py --out data/raw/web/tamilvu.jsonl \
  --state data/raw/web/tamilvu_state.json \
  --only-codes l1100,l1210,l1220,l1230,l1240,l1250,l1260,l1270,l1280,l1300,l2100,l3100,l3200,l3300,l3700,l3800,l0100,l0900 \
  --max-pages-per-book 400 --delay 0.12 --time-budget 2400

# whole catalogue
python scripts/ingest_tamilvu.py --max-pages-per-book 200 --time-budget 3600
```

Notes:
* Some works are not framed with `lft.jsp` (e.g. `arumpoorul`) and yield 0 pages;
  they need per-work handling.
* A few endpoints return `403` when crawled too fast — the crawler logs and
  continues (per-work failure isolation).
* Podcast/audio-only pages are skipped by the Tamil-text quality gate.

## Tamil Wikisource detail

Uses the MediaWiki API with `list=allpages` + `prop=extracts&explaintext=1`.
Requests are sent by **POST** because batches of long Tamil titles exceed the
URL length limit (HTTP 414).

```bash
python scripts/ingest_wikisource_ta.py --out data/raw/web/wikisource_ta.jsonl \
  --state data/raw/web/wikisource_ta_state.json --time-budget 1800 --min-chars 120
```

Many Wikisource pages are short subpages or template transclusions with empty
`extracts`; only pages above `--min-chars` and 50% Tamil are kept.

## Merging into the training corpus

The corpus expander reads a directory of `.jsonl` files. Point it at a staging
directory containing only the library outputs (so state `.json` files are not
mistaken for documents):

```bash
mkdir -p data/raw/weblibs
ln -sf ../web/tamilvu.jsonl data/raw/weblibs/tamilvu.jsonl
ln -sf ../web/wikisource_ta.jsonl data/raw/weblibs/wikisource_ta.jsonl

python scripts/expand_tamil_corpus_v2.py --resume --out-dir data/raw/expanded_v2 \
  --local-roots "$PWD/../open-sangam/data/texts,$PWD/../data/texts,$PWD/data/raw/weblibs" \
  --mc4-limit 0 --cc100-limit 0 --oscar-limit 0 --indic-limit 0 \
  --fineweb-limit 0 --news-limit 0 --books-limit 0

python scripts/expand_tamil_corpus_v2.py --recount-only --out-dir data/raw/expanded_v2
```

The expander seeds exact hashes from the existing shards, so re-running after a
longer crawl only appends genuinely new documents.

## Observed throughput

| Source | Pages fetched | Docs kept | Notes |
|---|---|---|---|
| tamilvu (priority 42 works) | ~900 | 850 | verses are dense; commentary opt-in |
| Tamil Wikisource | ~15k titles scanned | 77 | short subpages filtered out |

A full catalogue crawl is a multi-hour background job; run it with `setsid
nohup` and merge afterwards.

## Next sources to evaluate

* **Noolaham** — mostly scanned PDFs, requires OCR before it is useful here.
* **Tamil Ebooks / Free Tamil Ebooks** — HTML books, licence review needed.
* **Internet Archive Tamil texts** — OCR text layer available for many items.
* **Tamil Nadu government open data** — PDFs/reports; needs a PDF text extractor.

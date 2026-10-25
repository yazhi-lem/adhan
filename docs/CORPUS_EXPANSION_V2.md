# Adhan Corpus Expansion V2 — Target: 1.5–2 Billion Clean Swaram Tokens

> **Current state**: 328M tokens (1.33M docs, 2.9GB)  
> **Target state**: 1.5–2B tokens (~5–7M docs, ~15GB)  
> **Status**: Ready to run

---

## What's New in V2

The V1 pipeline (`scripts/expand_tamil_corpus.py`) gave us a solid 328M-token foundation from:
- Local classical texts (Open Sangam, Sangam literature)
- Tamil Wikipedia (150K docs)
- AI4Bharat IndicCorp v2 (1M docs)
- FineWeb-2 Tamil (157K docs)

**V2 adds 5 major new sources** and raises limits on existing ones:

| Source | V1 Limit | V2 Limit | Est. Tokens | Quality |
|--------|----------|----------|-------------|---------|
| **IndicCorp v2** | 1M docs | **3M docs** | ~900M | ★★★☆ |
| **FineWeb-2** | 157K docs | **1.5M docs** | ~450M | ★★★☆ |
| **mC4 Tamil** | — | **2M docs** | ~600M | ★★☆☆ |
| **CC-100 Tamil** | — | **1M docs** | ~200M | ★★☆☆ |
| **OSCAR Tamil** | — | **500K docs** | ~150M | ★★☆☆ |
| **Tamil News** | — | **500K docs** | ~100M | ★★★★ |
| **Tamil Books** | — | **200K docs** | ~60M | ★★★★ |
| **Wikipedia** | 150K docs | **500K docs** | ~80M | ★★★★ |
| **Web scrape** (Madurai, TamilVU) | — | **100K docs** | ~30M | ★★★★★ |
| **TOTAL** | | | **~2.57B** | |

After deduplication and quality filtering, expect **1.5–2B clean tokens**.

---

## Quick Start

### Full Expansion (requires 50GB disk, 16GB RAM, fast internet)

```bash
cd repos/adhan
source .venv/bin/activate

# Run everything at max limits
python scripts/expand_tamil_corpus_v2.py --full \
    --out-dir data/raw/expanded_v2

# This will take 4–8 hours depending on bandwidth
```

### Incremental — Add Only New Sources

```bash
# If you already have v1 corpus, seed dedup from it
python scripts/expand_tamil_corpus_v2.py \
    --mc4-limit 2000000 \
    --cc100-limit 1000000 \
    --oscar-limit 500000 \
    --news-limit 500000 \
    --books-limit 200000 \
    --seed-v1 data/raw/expanded \
    --out-dir data/raw/expanded_v2
```

### Web Scraping (Classical + Literary)

```bash
# Install dependencies
pip install requests beautifulsoup4 feedparser

# Scrape Project Madurai + Tamil Virtual Academy
python scripts/ingest_tamil_web.py --source all --out-dir data/raw/web

# Then merge into v2
python scripts/expand_tamil_corpus_v2.py \
    --local-roots data/raw/web \
    --out-dir data/raw/expanded_v2
```

---

## Quality Gates per Source

| Source | `min_chars` | `min_tamil_ratio` | Rationale |
|--------|-------------|-------------------|-----------|
| Local classical | 30 | 0.50 | Already curated |
| Wikipedia | 80 | 0.55 | Encyclopedia quality |
| Tamil News | 100 | 0.60 | Journalism standard |
| Tamil Books | 200 | 0.55 | Literary prose |
| IndicCorp v2 | 50 | 0.60 | Moderate upstream cleaning |
| FineWeb-2 | 60 | 0.60 | Language score + ratio |
| mC4 | 80 | 0.50 | Already cleaned by Google |
| CC-100 | 60 | 0.50 | Facebook extraction |
| OSCAR | 60 | 0.50 | Deduplicated crawl |
| Web scrape | 40 | 0.50 | Paragraph extraction |

---

## Deduplication Strategy

V2 uses the same **banded LSH** as V1 but with enhancements:

1. **Cross-version dedup**: Seeded with all documents from v1 corpus, so v2 never emits a duplicate of a v1 document
2. **Higher threshold**: 0.82 Jaccard (was 0.80) — slightly stricter to handle the noisier mC4/CC-100 sources
3. **Exact + near-dup**: SHA-1 exact match catches identical documents; LSH catches near-duplicates

Expected dedup rate:
- mC4 ↔ CC-100: ~15–20% overlap
- mC4 ↔ OSCAR: ~10–15% overlap
- FineWeb-2 ↔ CC-100: ~5–10% overlap
- Overall: ~20–30% of new sources will be deduplicated against existing or each other

---

## Tokenizer Retraining

After expanding the corpus, retrain the swaram tokenizer:

```bash
python scripts/freeze_fast_tokenizer.py \
    --corpus data/raw/expanded_v2 \
    --out data/final/tamil_slm_v3 \
    --vocab-size 32000 \
    --seq-len 512 \
    --max-tokens 0  # use entire corpus
```

**Fertility target**: Maintain `< 1.15 tokens/akshara` on held-out Tamil text.

---

## Disk & Memory Budget

| Phase | Disk | RAM | Time |
|-------|------|-----|------|
| V2 full expansion | ~50 GB | ~12 GB | 4–8 hours |
| Web scraping | ~2 GB | ~2 GB | 2–4 hours |
| Tokenizer retrain | ~5 GB | ~8 GB | 30–60 min |
| Shard packing | ~30 GB | ~4 GB | 1–2 hours |
| **Total** | **~90 GB** | **~16 GB peak** | **8–16 hours** |

---

## Validation

After building, validate the expanded corpus:

```bash
python scripts/phase2_validate.py \
    --corpus data/raw/expanded_v2 \
    --max-records 100000 \
    --output data/raw/expanded_v2/validation_report.json
```

Check:
- Tamil character ratio > 0.65 median
- Fertility < 1.15
- No PII leakage (sample inspection)
- Per-source quality scores look reasonable

---

## Integration with Training

Update your training config:

```yaml
# src/adhan_slm/configs/adhan_slm_nano_cpu.yaml
data:
  shards: data/final/tamil_slm_v3
  version: "v3-2B-tokens"
```

Then launch pretraining:

```bash
python -m adhan_slm.training.train_jax \
    --config src/adhan_slm/configs/adhan_slm_nano_cpu.yaml \
    --device cpu
```

With 1.5B+ tokens, you can now train for **much longer schedules** without overfitting:

```yaml
train:
  max_steps: 100000  # was 4000 — now we have the data to support it
  warmup_steps: 2000
```

---

## Roadmap

| Phase | Target | Action |
|-------|--------|--------|
| **V2.1** | 2B tokens | Run full expansion, validate, retrain tokenizer |
| **V2.2** | 3B tokens | Add Tamil YouTube transcripts (CC), Tamil cinema subtitles |
| **V2.3** | 5B tokens | Partner with Tamil Nadu government for open data releases |
| **V3** | 10B tokens | Full CommonCrawl Tamil subset (custom extraction) |

---

*அறிவுக்கு எல்லை இல்லை — Knowledge knows no bounds.*

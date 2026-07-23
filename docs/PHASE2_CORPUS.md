# Phase 2: Corpus Curation at Pretraining Scale

**Status**: ✅ Infrastructure Ready  
**Target**: 300M-1B clean Tamil tokens for adhan-nano pretraining  
**Branch**: `claude/adhan-phase2-corpus-v1`

---

## Overview

Phase 2 builds the corpus foundation for Phase 3 (GPU pretraining). This guide covers:

1. **Data sourcing strategy** - Where to get 300M-1B tokens
2. **Processing pipeline** - Dedup, filter, validate
3. **Integration with yazhi projects** - Using vazhi, corpus-tamil, Sangam
4. **Quality assurance** - Validation and metrics
5. **Reproducibility** - Code SHA, data sheets, versioning

---

## Architecture

```
Data Sources (Yazhi Ecosystem + Modern + News + Educational)
    ↓
[1] INGEST
    ├─ yazhi-lem/vazhi (QA pairs)
    ├─ yazhi-lem/corpus-tamil (pre-curated)
    ├─ Sangam/PMWorks (classical literature)
    ├─ Wikipedia (modern encyclopedic)
    ├─ Reddit (conversational)
    ├─ News sites (journalism)
    └─ Output: Unified JSONL
    
    ↓
[2] DEDUPLICATE (MinHash/LSH)
    ├─ Exact match (SHA-256)
    ├─ Near-duplicates (Jaccard > 0.85)
    └─ Output: Deduplicated JSONL + report
    
    ↓
[3] FILTER
    ├─ Quality (length, punctuation, fertility)
    ├─ Language (Tamil ≥ 70%)
    ├─ PII (remove emails, phones, URLs)
    └─ Output: Filtered JSONL
    
    ↓
[4] VALIDATE & REPORT
    ├─ Token count
    ├─ Fertility measurement
    ├─ PII spot-check
    └─ Output: Quality report + stats
    
    ↓
[5] PACK (via prepare_slm_corpus.py)
    ├─ Tokenize with swaram
    ├─ Pack into sequences
    └─ Output: train.bin, val.bin, test.bin
```

---

## Data Sourcing Strategy

### Priority 1: Yazhi Ecosystem (5-15% of corpus)

Your projects have high-quality Tamil data:

#### yazhi-lem/vazhi (QA Dataset)
- **Type**: Knowledge/QA pairs
- **Quality**: High (0.75)
- **Volume**: ~50k-100k QA pairs (potential 10-50M tokens)
- **Format**: JSON/JSONL
- **Integration**: `VazhiImporter` in `src/data_scraper/yazhi_integrations/`

```python
from src.data_scraper.yazhi_integrations import VazhiImporter

importer = VazhiImporter("/path/to/yazhi-lem/vazhi")
for record in importer.import_from_repo():
    print(record)
    # {
    #     "id": "vazhi-qa-123",
    #     "text": "Question + Answer text",
    #     "source": "vazhi-qa",
    #     "quality_score": 0.75
    # }
```

#### yazhi-lem/corpus-tamil (Pre-curated Corpus)
- **Type**: Diverse Tamil texts
- **Quality**: High (0.85)
- **Volume**: ~10k-50k records (potential 2-50M tokens)
- **Formats**: JSONL, JSON, TXT
- **Integration**: `CorpusTamilImporter`

```python
from src.data_scraper.yazhi_integrations import CorpusTamilImporter

importer = CorpusTamilImporter("/path/to/yazhi-lem/corpus-tamil")
for record in importer.import_from_repo():
    print(record)
```

#### Sangam + Yazhi API (Classical Literature)
- **Type**: Classical Tamil texts (Thirukkural, etc.)
- **Quality**: Very High (0.9)
- **Volume**: ~1k-5k texts (potential 5-20M tokens)
- **Formats**: JSON, JSONL, TXT, XML
- **Integration**: `SangamImporter`

```python
from src.data_scraper.yazhi_integrations import SangamImporter

importer = SangamImporter()
for record in importer.import_from_open_sangam("/path/to/sangam"):
    print(record)
```

### Priority 2: Modern Conversational (15-30%)

Modern Tamil discussions and conversational text:

| Source | Volume | Quality | Integration |
|--------|--------|---------|-------------|
| Reddit (tamil, tamilnadu) | 100k-500k | 0.5-0.7 | Use existing scraper |
| Twitter/Nitter | 200k-1M | 0.3-0.5 | Filter heavily |
| News comments | 50k-200k | 0.6 | Extend scraper |
| Conversational phrases | ~211 | 0.7 | YAML database |

**Why**: Conversational text teaches natural language patterns.

### Priority 3: News & Journalism (15-30%)

Professional Tamil journalism:

| Source | Volume | Quality | Notes |
|--------|--------|---------|-------|
| tamil.samayam.com | 100k-500k | 0.7-0.8 | Modern news |
| tamil.oneindia.com | 50k-200k | 0.7 | Regional coverage |
| Tamil press releases | 10k-50k | 0.8 | Official statements |
| Tamil blogs | 50k-200k | 0.6-0.7 | Long-form content |

**Why**: News teaches formal Tamil, diverse topics.

### Priority 4: Educational & Literature (30-40%)

Structured, high-quality Tamil:

| Source | Volume | Quality | Notes |
|--------|--------|---------|-------|
| Wikipedia (Tamil) | 10k-50k | 0.8-0.9 | Encyclopedic |
| Project Madurai | 1k-5k | 0.9 | Classical literature |
| School textbooks | 10k-50k | 0.8 | Educational |
| Technical Tamil docs | 5k-20k | 0.8 | Code comments, APIs |

**Why**: Educational text improves model understanding and reasoning.

---

## Processing Pipeline

### Step 1: Ingest (Week 1-2)

Collect data from all sources into unified JSONL format:

```json
{
  "id": "unique-doc-id",
  "text": "Tamil text content",
  "source": "source-type",
  "url": "original-url-or-null",
  "quality_score": 0.75
}
```

**Command**:
```bash
python scripts/phase2_corpus_build.py \
  --yazhi-projects ~/yazhi-projects \
  --output data/raw/phase2 \
  --log-level INFO
```

**Output**: `data/raw/phase2/unified.jsonl` (~1.5M-5M records)

### Step 2: Deduplicate (Week 2-3)

Remove exact and near-duplicates using MinHash/LSH:

```bash
# Built-in to phase2_corpus_build.py, configurable:
python scripts/phase2_corpus_build.py \
  --yazhi-projects ~/yazhi-projects \
  --dedup-threshold 0.85 \
  --output data/raw/phase2
```

**Settings**:
- Exact match: SHA-256 hashing
- Near-duplicate: Jaccard > 0.85 (configurable)
- Per-source dedup: Optional for Reddit, Twitter

**Output**: Deduplicated JSONL, `dedup_report.json`

**Expected**: 10-20% of records removed (typical dedup rate)

### Step 3: Filter (Week 3)

Apply quality, language, and PII filters:

**Quality Filtering**:
```python
filter = CorpusFilter(
    min_length=20,        # min 20 chars
    max_length=10000,     # max 10k chars
    min_quality_score=0.5 # score ≥ 0.5
)
```

**Language Filtering**:
```python
# Keep only Tamil-dominant text (≥ 70% Tamil chars)
filter = CorpusFilter(min_tamil_fraction=0.7)
```

**PII Scrubbing**:
```python
# Anonymize: emails → [EMAIL], phones → [PHONE], URLs → [URL]
filter.scrub_pii(documents, anonymize_level="standard")
```

### Step 4: Validate (Week 4)

Check final corpus quality:

```bash
python scripts/phase2_validate.py \
  --corpus data/raw/phase2/unified.jsonl \
  --sample-size 1000 \
  --output validation_report.json
```

**Validation checks**:
- ✅ 300M-1B tokens (measure in prepare_slm_corpus.py)
- ✅ Fertility < 1.15 tokens/akshara
- ✅ No PII in spot-check (100 samples)
- ✅ Language mix: ≥ 70% Tamil
- ✅ Quality distribution: mean ≥ 0.65

### Step 5: Pack & Freeze (Week 5)

Prepare for pretraining via existing `prepare_slm_corpus.py`:

```bash
python scripts/prepare_slm_corpus.py \
  --corpus-jsonl data/raw/phase2/unified.jsonl \
  --output-dir data/final/phase2 \
  --vocab-size 12000 \
  --max-seq-length 1024
```

**Output**:
- `train.bin` (80%)
- `val.bin` (10%)
- `test.bin` (10%)
- `vocab.json` (frozen tokenizer)
- `merges.txt` (BPE merges)
- `datasheet.json` (metadata)

---

## Running the Pipeline

### Quick Start

```bash
# 1. Setup
export YAZHI_ROOT=~/yazhi-projects
mkdir -p data/raw/phase2

# 2. Build corpus
python scripts/phase2_corpus_build.py \
  --yazhi-projects $YAZHI_ROOT \
  --output data/raw/phase2

# 3. Validate
python scripts/phase2_validate.py \
  --corpus data/raw/phase2/unified.jsonl \
  --output data/raw/phase2/validation_report.json

# 4. Pack for training
python scripts/prepare_slm_corpus.py \
  --corpus-jsonl data/raw/phase2/unified.jsonl \
  --output-dir data/final/phase2
```

### Advanced Options

```bash
# Custom dedup threshold (higher = more lenient)
python scripts/phase2_corpus_build.py \
  --yazhi-projects $YAZHI_ROOT \
  --dedup-threshold 0.90

# Minimum Tamil fraction
python scripts/phase2_corpus_build.py \
  --yazhi-projects $YAZHI_ROOT \
  --min-tamil-fraction 0.8

# Skip PII scrubbing (for trusted sources)
python scripts/phase2_corpus_build.py \
  --yazhi-projects $YAZHI_ROOT \
  --skip-pii

# Validation with all records (slower)
python scripts/phase2_validate.py \
  --corpus data/raw/phase2/unified.jsonl \
  # No --sample-size = check all records
```

---

## Data Format

### Input Format (from importers)

```json
{
  "id": "source-type-index",
  "text": "Tamil text content here",
  "source": "vazhi-qa",
  "url": "https://example.com/article",
  "quality_score": 0.75
}
```

### After Deduplication

```json
{
  "id": "source-type-index",
  "text": "Tamil text (exact/near-duplicate removed)",
  "source": "vazhi-qa",
  "url": "https://example.com/article",
  "quality_score": 0.75
}
```

### After Filtering

```json
{
  "id": "source-type-index",
  "text": "Tamil text (filtered, PII scrubbed → [EMAIL] [PHONE] [URL])",
  "source": "vazhi-qa",
  "url": "https://example.com/article",
  "quality_score": 0.75
}
```

---

## Quality Metrics & Targets

| Metric | Target | Why | Check |
|--------|--------|-----|-------|
| Total tokens | 300M-1B | Sufficient for pretrain | `wc -c` on .bin files |
| Fertility | < 1.15 tok/akshara | Efficient tokenization | `prepare_slm_corpus.py` logs |
| Tamil fraction | ≥ 70% average | Language quality | `phase2_validate.py` report |
| Quality score | ≥ 0.65 mean | Text quality | Filter removes < 0.5 |
| Dedup rate | 10-20% removal | No over-dedup | `phase2_corpus_build.py` report |
| PII presence | < 1% (spot-check) | Privacy assured | Manual review of 100 samples |

---

## Key Code Modules

### Importers (`src/data_scraper/yazhi_integrations/`)

1. **VazhiImporter** - QA pairs from vazhi-lem/vazhi
   ```python
   importer = VazhiImporter("/path/to/vazhi")
   for record in importer.import_from_repo():
       # Process record
   ```

2. **CorpusTamilImporter** - Pre-curated corpus
   ```python
   importer = CorpusTamilImporter("/path/to/corpus-tamil")
   for record in importer.import_from_repo():
       # Process record
   ```

3. **SangamImporter** - Classical Tamil literature
   ```python
   importer = SangamImporter()
   for record in importer.import_from_open_sangam("/path/to/sangam"):
       # Process record
   ```

### Processing (`src/adhan_slm/data/`)

1. **TextDeduplicator** - MinHash/LSH dedup
   ```python
   dedup = TextDeduplicator(threshold=0.85)
   is_dup, dup_of = dedup.is_duplicate(text, doc_id)
   ```

2. **CorpusFilter** - Quality/language/PII filtering
   ```python
   filter = CorpusFilter(min_tamil_fraction=0.7)
   filtered, stats = filter.filter_quality(documents)
   filtered, stats = filter.filter_language(filtered)
   filtered, stats = filter.scrub_pii(filtered)
   ```

### Scripts (`scripts/`)

1. **phase2_corpus_build.py** - Main orchestration
   ```bash
   python scripts/phase2_corpus_build.py --yazhi-projects ~/yazhi
   ```

2. **phase2_validate.py** - Quality validation
   ```bash
   python scripts/phase2_validate.py --corpus data/raw/phase2/unified.jsonl
   ```

---

## Reproducibility

### Recording Metadata

All corpus building must record:

```json
{
  "phase2_datasheet": {
    "timestamp": "2026-07-23T12:00:00Z",
    "corpus_size_tokens": 500000000,
    "total_records": 2000000,
    "sources": {
      "vazhi-qa": 50000,
      "corpus-tamil": 30000,
      "sangam-classical": 5000,
      "wikipedia": 40000,
      "reddit": 300000,
      "news": 200000,
      "educational": 100000
    },
    "processing": {
      "dedup_threshold": 0.85,
      "dedup_removed_rate": 0.15,
      "min_tamil_fraction": 0.7,
      "min_quality_score": 0.5,
      "pii_scrubbing": "standard"
    },
    "code_sha": "d4c2617a...",  # git commit hash
    "prepare_slm_corpus_version": "prepare_slm_corpus.py SHA",
    "swaram_tokenizer_version": "frozen at vocab_size=12000"
  }
}
```

### Version Control

1. **Branch**: `claude/adhan-phase2-corpus-v1`
2. **Commits**:
   - Phase 2.1: Yazhi importers
   - Phase 2.2: Dedup + filters
   - Phase 2.3: Orchestration + validation
3. **Final Commit**: Includes `datasheet.json` with full reproducibility data

### Verification Script

Before Phase 3 (pretraining), verify:

```bash
#!/bin/bash

# Check corpus files exist
ls -lh data/final/phase2/*.bin

# Check token count
python -c "
import numpy as np
for split in ['train', 'val', 'test']:
    data = np.load(f'data/final/phase2/{split}.bin')
    print(f'{split}: {len(data):,} tokens')
"

# Check fertility
python scripts/prepare_slm_corpus.py --check-fertility data/final/phase2

# Validate no PII in outputs
grep -i "email\|phone\|@\|+91" data/final/phase2/validation.jsonl || echo "✅ No PII"
```

---

## Timeline

| Week | Milestone | Tasks |
|------|-----------|-------|
| **Week 1** | Yazhi integration | Setup importers, test on vazhi/corpus-tamil |
| **Week 2** | Collection at scale | Run all scrapers, ingest to unified.jsonl |
| **Week 3** | Processing | Dedup + filter, validate quality |
| **Week 4** | Validation | Spot-check PII, measure fertility, generate datasheet |
| **Week 5** | Ready for Phase 3 | Pack shards, verify MLflow registration |

---

## Troubleshooting

### High Dedup Rate (>30%)

**Symptom**: More than 30% of records removed as duplicates.

**Diagnosis**:
- Sources overlap significantly
- Dedup threshold too high (> 0.9)
- Check `dedup_report.json` for which sources have most overlap

**Fix**:
```bash
# Lower threshold to be more strict
python scripts/phase2_corpus_build.py --dedup-threshold 0.80
```

### Low Tamil Fraction (<60%)

**Symptom**: Many records filtered out for insufficient Tamil.

**Diagnosis**:
- Mixed-language sources (Reddit, Twitter)
- News sites with English titles

**Fix**:
```bash
# Adjust threshold
python scripts/phase2_corpus_build.py --min-tamil-fraction 0.6

# Or manually filter source
# Remove source="reddit" from unified.jsonl if too mixed
```

### High PII Rate (>5%)

**Symptom**: Many records contain emails/phones/URLs.

**Diagnosis**:
- Social media sources (Twitter, Reddit comments)
- News with contact info

**Fix**: PII is automatically scrubbed, just note in datasheet.

### Fertility > 1.15

**Symptom**: Tokenizer fertility exceeds target.

**Diagnosis**:
- Quality score low (many abbreviations, numbers)
- Check `validation_report.json` for details

**Fix**: Increase `min_quality_score` threshold.

---

## Next Steps (Phase 3)

Once Phase 2 corpus is ready:

1. ✅ Verify corpus statistics in datasheet
2. ✅ Check MLflow registration
3. ✅ Run fertility validation < 1.15
4. 🚀 **Start Phase 3 GPU pretrain**:
   ```bash
   python src/adhan_slm/training/train_jax.py \
     --config configs/pretrain/adhan-nano.yaml \
     --corpus-dir data/final/phase2
   ```

---

## References

- **Plan**: `/root/.claude/plans/do-gap-analysis-for-polished-milner.md`
- **Tracking**: `docs/COMPLETION_TRACKER.md`
- **Deployment**: `docs/DEPLOYMENT.md`
- **Roadmap**: `ROADMAP_JAX_SLM.md`

---

**Status**: 🟡 Phase 2 Infrastructure Ready - Awaiting Data Collection

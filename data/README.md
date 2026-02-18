# Data Directory - Final Structure

## Overview

The data directory is now organized into 3 logical sections:

```
data/
├── raw/              (Source data - original inputs)
├── intermediate/     (Pipeline working files - intermediate outputs)
└── final/            (Training-ready datasets - final outputs)
```

---

## 1. raw/ - Source Data

**Purpose**: Original source materials and extracted data files from various sources.

**Contents**:
- `tamil_social_sample.jsonl` — Tamil social media samples (conversational)
- `tamil_corpus.txt` — Generic Tamil text corpus
- `projectmadurai_manifests/` — Project Madurai metadata and extracted content
- `pdf_books_manifests/` — PDF book extracted content (Project Madurai, TamilVu)
- `projectmadurai_hf/` — HF format intermediate (for reference)
- `tamilvu_manifests/`, `tamilvu_texts/` — TamilVu source data
- `raw_html/`, `raw_pdf/`, `pdf_texts/` — Raw extracted files

**Size**: ~268 MB

**Usage**: 
- Source input for data pipeline scripts
- Used by `src/data_scraper/raw_extractors/` modules
- Do NOT modify directly for training

---

## 2. intermediate/ - Pipeline Working Files

**Purpose**: Files generated during data processing pipeline stages.

**Structure**:

### 2a. intermediate/sentences/ - Sentence Extraction Outputs
```
wiki_sentences.jsonl           (Local sources: 1,493 sentences)
wiki_api_sentences.jsonl       (Wikipedia API: 1,427 sentences)
```

**What**: Output from sentence extraction scripts
- `extract_wiki_sentences.py` → wiki_sentences.jsonl
- `wikipedia_api_extractor.py` → wiki_api_sentences.jsonl

**Usage**: 
- Intermediate step in pipeline
- Merged to create v1_original.jsonl
- Do NOT use directly for training

---

### 2b. intermediate/rebalancing/ - Processing Versions
```
v1_original.jsonl              (2,918 sentences - merged from wiki sources)
v2_rebalanced.jsonl            (2,900 sentences - quality filtered with weighting)
v3_modern_enhanced.jsonl       (3,066 sentences - modern sources added)
```

**What**: 
- `v1_original` = Output from `merge_hf_datasets.py` (all sources merged, deduplicated)
- `v2_rebalanced` = Output from `build_modern_tamil_corpus.py` (reweighted for quality)
- `v3_modern_enhanced` = Output from `build_modern_tamil_sources.py` (modern sources added)

**Usage**:
- For analysis and comparison
- `v3_modern_enhanced` is the best version before HF export
- Do NOT use directly for training (use `final/` instead)

---

## 3. final/ - Training-Ready Datasets

**Purpose**: Final, processed datasets ready for model training.

**Structure**:
```
final/tamil_texts/hf/
├── train.jsonl              (1,220 records - 80%)
├── validation.jsonl         (152 records - 10%)
├── test.jsonl               (154 records - 10%)
└── README.md                (Dataset documentation)
```

**What**:
- Output from `export_modern_hf.py`
- HuggingFace-compatible format
- Stratified train/val/test splits (80/10/10)
- Ready for immediate training

**Format** (JSONL - one record per line):
```json
{
  "id": "unique_hash",
  "text": "Tamil text content...",
  "source": "wikipedia|news|social|local|literature",
  "quality_score": 0.524,
  "tamil_fraction": 0.95,
  "url": "source_url_or_null"
}
```

**Size**:
- train.jsonl: ~2.2 MB (1,220 records)
- validation.jsonl: ~275 KB (152 records)
- test.jsonl: ~280 KB (154 records)
- Total: ~2.75 MB

**Dataset Statistics**:
- Total records: 1,526
- Avg quality score: 0.524/1.0
- Tamil coverage: ~85%
- Modern sources: 9.8%

**✅ USE THIS FOR TRAINING!**

---

## Pipeline Flow

```
raw/
  ├─ projectmadurai_manifests/
  ├─ pdf_books_manifests/
  ├─ tamil_social_sample.jsonl
  └─ tamil_corpus.txt
           ↓
    [Data Extraction Scripts]
           ↓
intermediate/sentences/
  ├─ wiki_sentences.jsonl (local sources)
  └─ wiki_api_sentences.jsonl (Wikipedia API)
           ↓
    [Merge & Rebalance Scripts]
           ↓
intermediate/rebalancing/
  ├─ v1_original.jsonl (merged, deduplicated)
  ├─ v2_rebalanced.jsonl (quality filtered)
  └─ v3_modern_enhanced.jsonl (modern sources added)
           ↓
    [HF Export Script]
           ↓
final/tamil_texts/hf/ ← USE FOR TRAINING
  ├─ train.jsonl (80%)
  ├─ validation.jsonl (10%)
  └─ test.jsonl (10%)
```

---

## Using the Data in Training

### Notebook 1: Setup & Exploration

```python
from pathlib import Path
DATA_DIR = Path('data/final/tamil_texts/hf')

train_file = DATA_DIR / 'train.jsonl'
val_file = DATA_DIR / 'validation.jsonl'
test_file = DATA_DIR / 'test.jsonl'
```

### Notebook 2: Training

```python
# Load HF dataset
from datasets import load_dataset

dataset = load_dataset(
    'json',
    data_files={
        'train': 'data/final/tamil_texts/hf/train.jsonl',
        'validation': 'data/final/tamil_texts/hf/validation.jsonl',
        'test': 'data/final/tamil_texts/hf/test.jsonl',
    }
)
```

---

## Key Statistics

### Modern Tamil Corpus (v3)
```
Total Records: 3,066
├─ Wikipedia: 1,425 (46.5%)
├─ Local (Project Madurai): 1,267 (41.3%)
├─ News: 269 (8.8%)
├─ Literature: 75 (2.4%)
├─ Social: 20 (0.7%)
└─ Modern Conversational: 10 (0.3%)

Final Train/Val/Test Split: 1,220 / 152 / 154 records
Quality Avg: 0.524/1.0
Modern Sources: 9.8%
```

### Improvements from v1 to v3
```
Original (v1):          3.1% modern sources
Modern-Enhanced (v3):   9.8% modern sources
Improvement:            +6.7 percentage points (+180%)

Quality (news):         0.486 → 0.531 (+0.045)
Quality (social):       0.359 → 0.492 (+0.133)
```

---

## Maintenance & Updates

### To Update Training Data

1. **Add new sources** → Save to `raw/`
2. **Run extraction** → Scripts write to `intermediate/sentences/`
3. **Merge & rebalance** → Scripts create `intermediate/rebalancing/v*`
4. **Export to HF** → Script writes to `final/tamil_texts/hf/`

### To Analyze Pipeline

1. Compare versions:
   ```bash
   wc -l data/intermediate/rebalancing/*.jsonl
   ```

2. Show distribution:
   ```python
   import json
   # Load and analyze any intermediate file
   ```

3. Validate final dataset:
   ```bash
   # Check record count
   wc -l data/final/tamil_texts/hf/train.jsonl
   ```

---

## Not Recommended

❌ Do NOT:
- Manually edit JSONL files (breaks records)
- Use intermediate/ files directly for training (use final/)
- Delete files in final/tamil_texts/hf/ without backup
- Modify raw/ source files

✅ DO:
- Use data from `final/tamil_texts/hf/` for training
- Keep backup of `intermediate/rebalancing/v3_modern_enhanced.jsonl`
- Add new raw sources and re-run pipeline
- Document any custom modifications

---

## Size Reference

```
raw/                         268 MB (source data, kept for reproducibility)
intermediate/                  4 MB (working files, can be regenerated)
final/                       2.7 MB (training data, production)
────────────────────────────────────────
Total data/                  275 MB

Breakdown by purpose:
- Sources (raw): 268 MB  (kept as reference)
- Pipeline (int): 4 MB   (working temp files, regenerable)
- Training (fin): 2.7 MB (final, most important)
```

---

**Last Updated**: February 19, 2026  
**Status**: ✅ Final structure verified  
**Training Data Location**: `data/final/tamil_texts/hf/`  
**Next Step**: Use in notebooks for training

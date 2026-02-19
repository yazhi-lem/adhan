# Data Scraper Export Scripts

## Consolidated Scripts

### export_unified_hf.py (NEW - Recommended)

**Replaces**: `export_hf_from_sentences.py` and `export_modern_hf.py`

Unified HuggingFace dataset exporter with configurable weighting strategies.

**Usage:**

```bash
# Standard export (uniform weighting)
python src/data_scraper/export/export_unified_hf.py \
    --input data/pre_training/tamil_texts/all_sentences.jsonl \
    --output data/pre_training/tamil_texts/hf \
    --strategy standard

# Modern export (boost contemporary sources)
python src/data_scraper/export/export_unified_hf.py \
    --input data/pre_training/tamil_texts/all_sentences_modern.jsonl \
    --output data/pre_training/tamil_texts/hf \
    --strategy modern \
    --modern-ratio 0.60
```

**Options:**
- `--input`: Input JSONL file path (required)
- `--output`: Output directory for HF splits (default: data/pre_training/tamil_texts/hf)
- `--strategy`: Weighting strategy - `standard` or `modern` (default: standard)
- `--modern-ratio`: Target ratio for modern sources when using modern strategy (default: 0.60)
- `--train-ratio`: Train split ratio (default: 0.8)
- `--val-ratio`: Validation split ratio (default: 0.1)
- `--seed`: Random seed for shuffling (default: 42)
- `--no-dedupe`: Skip deduplication step

**Features:**
- Configurable weighting strategies
- Stratified sampling for modern strategy
- Automatic deduplication
- Deterministic train/val/test splits
- Detailed statistics and README generation

## Legacy Scripts (Deprecated - To Be Removed)

### export_hf_from_sentences.py
- Basic export with 80/10/10 split
- **Use `export_unified_hf.py --strategy standard` instead**

### export_modern_hf.py
- Export with modern source boosting
- **Use `export_unified_hf.py --strategy modern` instead**

## Migration Guide

**Before:**
```bash
python src/data_scraper/export/export_hf_from_sentences.py
# OR
python src/data_scraper/export/export_modern_hf.py
```

**After:**
```bash
# For standard export (replaces export_hf_from_sentences.py)
python src/data_scraper/export/export_unified_hf.py \
    --input data/pre_training/tamil_texts/all_sentences.jsonl \
    --strategy standard

# For modern export (replaces export_modern_hf.py)
python src/data_scraper/export/export_unified_hf.py \
    --input data/pre_training/tamil_texts/all_sentences_modern.jsonl \
    --strategy modern
```

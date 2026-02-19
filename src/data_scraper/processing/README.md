# Data Scraper Processing Scripts

## Consolidated Scripts

### build_unified_corpus.py (NEW - Recommended)

**Replaces**: `build_modern_tamil_corpus.py` and `build_modern_tamil_sources.py`

Unified corpus builder with configurable modern/classical balancing.

**Usage:**

```bash
# Build with modern strategy (prioritize contemporary sources)
python src/data_scraper/processing/build_unified_corpus.py \
    --data-dir data/raw \
    --existing-corpus data/pre_training/tamil_texts/all_sentences.jsonl \
    --output data/pre_training/tamil_texts/all_sentences_modern.jsonl \
    --strategy modern

# Build with balanced strategy (uniform weighting)
python src/data_scraper/processing/build_unified_corpus.py \
    --strategy balanced \
    --output data/pre_training/tamil_texts/all_sentences_balanced.jsonl

# Build with only modern sources (no existing corpus)
python src/data_scraper/processing/build_unified_corpus.py \
    --strategy modern \
    --modern-only \
    --output data/pre_training/tamil_texts/modern_only.jsonl

# Rebalance with size limit
python src/data_scraper/processing/build_unified_corpus.py \
    --strategy rebalanced \
    --max-records 2900 \
    --output data/pre_training/tamil_texts/all_sentences_rebalanced.jsonl
```

**Options:**
- `--data-dir`: Directory containing raw data files (default: data/raw)
- `--existing-corpus`: Path to existing corpus JSONL file (default: data/pre_training/tamil_texts/all_sentences.jsonl)
- `--output`: Output JSONL file path (default: data/pre_training/tamil_texts/all_sentences_modern.jsonl)
- `--strategy`: Corpus building strategy - `balanced`, `modern`, or `rebalanced` (default: modern)
- `--max-records`: Maximum number of records to keep (default: None = all)
- `--modern-only`: Only use modern sources, ignore existing corpus

**Strategies:**

1. **balanced**: Uniform weighting across all sources
   - Equal priority for all source types
   - No boosts or penalties

2. **modern**: Prioritize contemporary Tamil
   - News: 3.0x weight
   - Social: 2.5x weight
   - Modern conversational: 3.0x weight
   - Wikipedia: 1.5x weight
   - Literature: 0.5x weight
   - Classical/local: 0.3x weight

3. **rebalanced**: Similar to modern but tuned for quality
   - News: 3.0x weight
   - Social: 2.5x weight
   - Modern conversational: 2.5x weight
   - Wikipedia: 1.0x weight
   - Literature: 0.5x weight
   - Classical/local: 0.3x weight

**Features:**
- Automatic modern marker detection
- Source-based weighting
- Quality scoring
- Deduplication
- Detailed statistics

## Legacy Scripts (Deprecated - To Be Removed)

### build_modern_tamil_corpus.py
- Rebalances corpus by source weighting
- Filters archaic texts
- **Use `build_unified_corpus.py --strategy rebalanced` instead**

### build_modern_tamil_sources.py
- Builds corpus from modern sources
- Adds conversational phrases
- **Use `build_unified_corpus.py --strategy modern` instead**

## Other Processing Scripts

### build_pretraining_sentences.py
- Comprehensive sentence building from multiple sources
- Includes colloquial detection
- **Still in use - no replacement needed**

### extract_wiki_sentences.py
- Wikipedia-specific sentence extraction
- **Still in use - no replacement needed**

### merge_hf_datasets.py
- Combines multiple HuggingFace datasets
- **Still in use - no replacement needed**

## Migration Guide

**Before:**
```bash
python src/data_scraper/processing/build_modern_tamil_corpus.py
# OR
python src/data_scraper/processing/build_modern_tamil_sources.py
```

**After:**
```bash
# For rebalancing (replaces build_modern_tamil_corpus.py)
python src/data_scraper/processing/build_unified_corpus.py \
    --strategy rebalanced \
    --max-records 2900

# For modern sources (replaces build_modern_tamil_sources.py)
python src/data_scraper/processing/build_unified_corpus.py \
    --strategy modern
```

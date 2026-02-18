# Adhan: Yazhi's first Tamil LLM Model

Our own Tamil First GPT-based LLM Model.  
Powered by Sangam Literature and rooted in the earliest Tamil words — from Adichanallur to Keezhadi.

## The Name

**Adhan** (ஆதன்) comes from **ஆதி** (Aadhi) — meaning **beginning** or **first** — honoring the earliest written Tamil discovered at ancient archaeological sites.

> *Just as the first Tamil words were etched on pottery thousands of years ago,  
> Adhan carries that legacy into the AI age.*

---

## Project Overview

**Adhan** is a fully open-source initiative to build a modern, high-quality Tamil language model (LLM) and a reproducible, modular pipeline for Tamil NLP research and production. The project is designed for the FOSS (Free and Open Source Software) community, with a focus on transparency, extensibility, and real-world usability.

---

## Key Features

- **Open Data Pipeline**: All scripts for scraping, cleaning, and processing Tamil text from diverse sources (literature, Wikipedia, news, social media).
- **Modern Corpus**: Balanced, deduplicated, and quality-scored dataset with a strong emphasis on contemporary, colloquial Tamil.
- **Reproducible Training**: Notebooks and scripts for end-to-end masked language model (MLM) training using Hugging Face Transformers.
- **Extensible**: Add new sources, preprocessing, or downstream tasks with minimal changes.
- **Documentation**: Comprehensive guides, quick references, and API docs for every module.

---

## Directory Structure

```
data/
  ├── raw/           # Raw HTML, PDF, manifests, and samples
  │   └── .cache/    # Scraper cache (auto-generated)
  ├── intermediate/  # Pipeline working files (extracted, rebalanced, etc.)
  └── final/         # Training-ready splits (Hugging Face format)
src/
  ├── data_scraper/  # Extraction, processing, export, and evaluation scripts
  │   ├── raw_extractors/       # Web scrapers
  │   │   ├── tamil_corpus_scraper_enhanced.py  # ✨ NEW: Enhanced scraper with caching
  │   │   ├── tamil_corpus_scraper.py           # Legacy scraper
  │   │   ├── wikipedia_api_extractor.py
  │   │   ├── pdf_book_scraper.py
  │   │   └── pmworks_extractor.py
  │   ├── processing/           # Data processing
  │   │   ├── build_unified_corpus.py           # ✨ NEW: Unified corpus builder
  │   │   ├── build_pretraining_sentences.py
  │   │   └── extract_wiki_sentences.py
  │   ├── export/               # HuggingFace format export
  │   │   └── export_unified_hf.py              # ✨ NEW: Unified exporter
  │   └── evaluation/           # Quality validation
  └── models/
      └── sangam_gpt/
          ├── train_enhanced.py                 # ✨ NEW: Enhanced training with config
          ├── train.py                          # Legacy training script
          └── config.yaml                       # Training configuration
  └── notebooks/     # Jupyter notebooks for exploration and training
```

**✨ New Features:**
- **Enhanced Scraper**: Improved error handling, caching, and retry logic
- **Unified Scripts**: Consolidated duplicate functionality
- **Config-based Training**: YAML/JSON configuration support
- **Pipeline Integration**: Seamless data → training workflow


---

## Specifications

- **Language**: Tamil (Unicode, TSCII/Mylai conversion supported)
- **Corpus**: Wikipedia, Project Madurai, news, social, and more
- **Preprocessing**: Deduplication, sentence segmentation, quality/colloquial scoring, rebalancing
- **Export**: HuggingFace-style train/val/test splits (`.jsonl`)
- **Training**: Hugging Face Transformers (notebooks provided)
- **Model**: XLM-RoBERTa-base (124M params, 10 epochs, 32 batch, 5e-5 LR)
- **Tokenization**: 250K vocab, 512 max length
- **Environment**: Python 3.11, PyTorch 2.1+, transformers 4.35+

---

## Quickstart

1. **Clone the repo**
   ```bash
   git clone https://github.com/yourusername/adhan.git
   cd adhan
   ```
2. **Install dependencies**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
3. **Prepare data**
   - Place raw data in `data/raw/documents/` or run scrapers in `src/data_scraper/`.
   - Run extraction and cleaning scripts as needed.
4. **Train the model**
   - Use notebooks in `src/notebooks/` (e.g., `01_setup_and_exploration.ipynb`, `02_model_training.ipynb`).
   - Training-ready data is in `data/final/tamil_texts/hf/`.

---

## Usage

### Complete End-to-End Pipeline

**1. Scrape and collect Tamil text data:**
```bash
# Use enhanced scraper with caching
python src/data_scraper/raw_extractors/tamil_corpus_scraper_enhanced.py \
    --source wikipedia \
    --category Tamil_language \
    --max-articles 50 \
    --output tamil_corpus_enhanced.jsonl
```

**2. Build modern Tamil corpus:**
```bash
# Build corpus with modern source prioritization
python src/data_scraper/processing/build_unified_corpus.py \
    --strategy modern \
    --output data/pre_training/tamil_texts/all_sentences_modern.jsonl
```

**3. Export to HuggingFace format:**
```bash
# Export with modern weighting strategy
python src/data_scraper/export/export_unified_hf.py \
    --input data/pre_training/tamil_texts/all_sentences_modern.jsonl \
    --output data/pre_training/tamil_texts/hf \
    --strategy modern
```

**4. Train the model:**
```bash
# Create configuration file
python src/models/sangam_gpt/train_enhanced.py --create-config config.yaml

# Edit config.yaml to customize settings, then train
python src/models/sangam_gpt/train_enhanced.py --config config.yaml
```

### Alternative: Legacy Scripts

- **Extract and clean data:**
  ```bash
  python src/data_scraper/pmworks_extractor.py
  python src/data_scraper/wikipedia_api_extractor.py
  ```
- **Train a model (from notebook):**
  Open and run cells in `src/notebooks/02_model_training.ipynb`.
- **Export to Hugging Face Datasets:**
  ```bash
  python src/data_scraper/export/export_hf_from_sentences.py  # DEPRECATED
  ```

**Note**: Legacy scripts are deprecated. Please use the new unified scripts for better reliability and features.


---

## Community & Contribution

- **FOSS**: 100% open code, data, and documentation
- **Contributions**: PRs, issues, and new modules welcome
- **License**: MIT

---

## Project Status

- ✅ Data pipeline refactored and documented
- ✅ Modern Tamil corpus created and analyzed
- ✅ Training-ready splits available
- ✅ Notebooks for setup, exploration, and training
- ✅ All data and scripts open for the community

---

## See Also

- [ADHAN_STORY.md](./ADHAN_STORY.md) — The full renaming story
- [PROJECT_CONFIG.md](./PROJECT_CONFIG.md) — Project configuration
- [TRAINING_GUIDE.md](./TRAINING_GUIDE.md) — Training documentation

---

*ஆதன் — தமிழ் முதல்*

# Adhan: FOSS Tamil LLM Project

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
  ├── intermediate/  # Pipeline working files (extracted, rebalanced, etc.)
  └── final/         # Training-ready splits (Hugging Face format)
src/
  ├── data_scraper/  # Extraction, processing, export, and evaluation scripts
  └── notebooks/     # Jupyter notebooks for exploration and training
```

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

- **Extract and clean data:**
  ```bash
  python src/data_scraper/pmworks_extractor.py
  python src/data_scraper/wikipedia_api_extractor.py
  # ...other scripts as needed
  ```
- **Train a model (from notebook):**
  Open and run cells in `src/notebooks/02_model_training.ipynb`.
- **Export to Hugging Face Datasets:**
  ```bash
  python src/data_scraper/export_hf_from_sentences.py
  ```

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

# Projects Configuration

## Project: Aadhan
- **Description:** Aadhan (Tamil: ஆதான்) - Foundational models and datasets for Tamil language processing.
- **Location:** `/home/neutron/.openclaw/zorba/Projects/OSS/yazhi/models/aadhan/`
- **Type:** Open Source Software (OSS)
- **Timestamp:** 2026-02-17 21:34:00
- **Key Paths:**
  - `src/` - Source code directory
  - `data/` - Data processing scripts and datasets
  - `docs/` - Documentation
  - `models/` - Trained models and configurations
  - `README.md` - Project overview

## Important Scripts
- `src/data_scraper/` - Web scraping and data collection tools
- `src/models/sangam_gpt/train.py` - Sangam GPT training pipeline
- `src/data_scraper/pdf_book_scraper.py` - PDF book processing
- `src/data_scraper/tamil_corpus_scraper.py` - Tamil corpus collection

## Dependencies
- Python 3.8+
- Hugging Face Transformers
- PyTorch/TensorFlow
- Web scraping libraries (BeautifulSoup, Scrapy)

## Setup Instructions
1. Clone the repository: `git clone https://github.com/yazhi-lem/aadhan.git`
2. Install dependencies: `pip install -r requirements.txt`
3. Configure data paths in `src/config.py`
4. Run data collection: `python src/data_scraper/run_all.py`
5. Train models: `python src/models/sangam_gpt/train.py`

## GitHub Integration
- Remote: `https://github.com/yazhi-lem/aadhan`
- Branch: `main`
- Authentication: HTTPS with GitHub PAT (configure via `git config --global credential.helper`)

## Project Status
- **Initial Commit:** Complete
- **Data Collection:** In progress
- **Model Training:** Not started
- **Documentation:** Basic structure created

## Next Steps
1. Configure GitHub credentials for automated pushes
2. Set up CI/CD pipeline
3. Create issue templates matching Capitol task types
4. Integrate with Capitol task management system
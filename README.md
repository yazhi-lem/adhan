# Adhan

Tamil-first LLM data + training pipeline.

## Recent changes

- Added shared constants in `src/core/`
- Added corpus merger: `src/data_scraper/merge_corpora.py`
- Added Gemma training notebook: `src/notebooks/03_gemma_training.ipynb`

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run scripts

Use dedicated scripts:

- `scripts/run_scraper.py` for corpus build + HF export
- `scripts/run_training.py` for training
- `scripts/run_model.py` for full orchestration

```bash
# Build corpus + export HF splits
python scripts/run_scraper.py --strategy modern --max-records 80000

# Train model
python scripts/run_training.py --num-epochs 3 --batch-size 4

# Full run (build + train, optional merge)
python scripts/run_model.py --strategy modern --num-epochs 3 --batch-size 4
```

For full command sequence and examples, see `DEV.md`.

## Core scripts

- `src/data_scraper/processing/build_unified_corpus.py`
- `src/data_scraper/export/export_unified_hf.py`
- `src/data_scraper/merge_corpora.py`
- `src/models/sangam_gpt/train_enhanced.py`
- `scripts/run_scraper.py`
- `scripts/run_training.py`
- `scripts/run_model.py`

## License

MIT

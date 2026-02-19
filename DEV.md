# DEV Commands

Minimal command sequence for local development.

## 1) Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2) Build corpus (dependency order)

```bash
python scripts/run_scraper.py \
  --strategy modern \
  --data-dir data/raw \
  --existing-corpus data/intermediate/rebalancing/v3_modern_enhanced.jsonl \
  --corpus-output data/intermediate/rebalancing/unified_modern.jsonl \
  --hf-output data/final/tamil_texts/hf \
  --max-records 80000
```

## 3) Train

```bash
python scripts/run_training.py \
  --data-dir data/final/tamil_texts/hf \
  --output-dir models/adhan \
  --num-epochs 3 \
  --batch-size 4 \
  --learning-rate 5e-5
```

## 4) Optional merge (ADHAN + VAZHI)

```bash
python src/data_scraper/merge_corpora.py \
  --adhan_dir data/final/tamil_texts/hf \
  --vazhi_repo ../vazhi \
  --output data/unified/tamil_6k.jsonl \
  --target_count 6000 \
  --split
```

## 5) One command (build + train)

```bash
python scripts/run_model.py \
  --strategy modern \
  --max-records 80000 \
  --num-epochs 3 \
  --batch-size 4
```

## Notes

- Run from repository root.
- Use `python scripts/run_scraper.py --help`, `python scripts/run_training.py --help`, and `python scripts/run_model.py --help`.
- Current runner executes:
  - `build_unified_corpus.py`
  - `export_unified_hf.py`
  - `train_enhanced.py`
  - `src/data_scraper/merge_corpora.py` (optional)

# REPORT

Date: 2026-02-19
Scope: Notebook flow review + project execution progress snapshot

## Notebook Flow

### 1) `src/notebooks/01_setup_and_exploration.ipynb`
- Purpose: Environment setup, corpus exploration, tokenizer prep, tokenization.
- Flow:
  1. Install/import dependencies
  2. Configure project/data paths
  3. Load HF JSONL splits and inspect distribution
  4. Visualize quality/source/length/tamil-fraction metrics
  5. Load tokenizer (`xlm-roberta-base`)
  6. Convert to datasets and tokenize
  7. Save tokenized datasets to `models/tokenized_datasets/`
- Status note: Notebook metadata shows cells not executed in current session, but some stored outputs/errors exist.

### 2) `src/notebooks/02_model_training.ipynb`
- Purpose: MLM pretraining pipeline with XLM-RoBERTa.
- Flow:
  1. Load tokenized datasets from disk
  2. Load model/tokenizer
  3. Configure MLM collator (15% masking)
  4. Set `TrainingArguments`
  5. Initialize `Trainer`
  6. Train
  7. Evaluate on validation + test
  8. Save final model (`models/aadhan-mlm-v1/`)
  9. Run mask-fill sanity tests
  10. Plot metrics
- Status note: Notebook metadata shows cells not executed in current session, with some persisted error outputs.

### 3) `src/notebooks/03_gemma_training.ipynb`
- Purpose: QLoRA fine-tuning of Gemma 3 1B-it + save adapter + GGUF instructions.
- Flow:
  1. Setup, paths, hyperparameters
  2. Load data and tokenize
  3. 4-bit model load + LoRA adapter apply
  4. Train with `Trainer`
  5. Evaluate
  6. Save adapter/tokenizer/config
  7. GGUF conversion instructions (`llama.cpp`)
- Status note: Notebook metadata shows cells not executed in current session.

## Progress Snapshot

- Project now has split operational runners:
  - `scripts/run_scraper.py`
  - `scripts/run_training.py`
  - `scripts/run_model.py`
- Corpus merge utility moved to:
  - `src/data_scraper/merge_corpora.py`
- Docs updated:
  - `README.md`
  - `DEV.md`

## Issues Observed During Review

1. Data split naming mismatch in Gemma notebook:
   - `03_gemma_training.ipynb` expects `val.jsonl`
   - Export pipeline writes `validation.jsonl`
2. Inconsistent status wording in notebooks:
   - Notebooks show “not executed” metadata for current session, but contain persisted old outputs/errors.
3. `02_model_training.ipynb` references a next notebook `03_fine_tuning_tasks.ipynb`, but current third notebook is `03_gemma_training.ipynb`.

## Next Steps (Recommended)

1. Standardize split filename usage to one convention:
   - Prefer `train.jsonl`, `validation.jsonl`, `test.jsonl` across scripts + notebooks.
2. Sync notebook references and summaries:
   - Update `02_model_training.ipynb` final section to point to `03_gemma_training.ipynb`.
3. Add one reproducible run path in docs (already mostly done):
   - Use `scripts/run_model.py` for end-to-end.
4. Optional cleanup pass:
   - Clear stale outputs in notebooks before sharing/release.

## Immediate Command Sequence

```bash
# 1) Build corpus + HF splits
python scripts/run_scraper.py --strategy modern --max-records 80000

# 2) Train model
python scripts/run_training.py --data-dir data/final/tamil_texts/hf --num-epochs 3 --batch-size 4

# 3) Full orchestrated run
python scripts/run_model.py --strategy modern --max-records 80000 --num-epochs 3 --batch-size 4
```

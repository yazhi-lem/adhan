# Tamil Model Training

Enhanced training system for Tamil language models with data pipeline integration.

## Quick Start

### 1. Using Configuration File (Recommended)

```bash
# Create default configuration
python src/models/sangam_gpt/train_enhanced.py --create-config config.yaml

# Edit config.yaml to customize settings

# Train with config file
python src/models/sangam_gpt/train_enhanced.py --config config.yaml
```

### 2. Using Command-Line Arguments

```bash
python src/models/sangam_gpt/train_enhanced.py \
    --model-name "sangam/IndianLanguages-Tamil-BERT-v0.1" \
    --data-dir "data/pre_training/tamil_texts/hf" \
    --output-dir "models/tamil_model" \
    --num-epochs 3 \
    --batch-size 4 \
    --learning-rate 5e-5
```

### 3. With Weights & Biases Tracking

```bash
# First install wandb: pip install wandb
# Login: wandb login

python src/models/sangam_gpt/train_enhanced.py \
    --config config.yaml \
    --use-wandb
```

## Configuration Options

### Model Settings

- `model_name`: Pretrained model name or path
- `model_type`: `causal_lm` (GPT-style) or `masked_lm` (BERT-style)
- `vocab_size`: Custom vocabulary size (optional)
- `max_length`: Maximum sequence length (default: 512)

### Data Settings

- `data_dir`: Directory containing train/val/test JSONL files
- `train_file`: Training data filename (default: train.jsonl)
- `val_file`: Validation data filename (default: validation.jsonl)
- `test_file`: Test data filename (default: test.jsonl)
- `text_column`: Column name containing text (default: text)

### Training Settings

- `output_dir`: Directory to save trained model
- `num_epochs`: Number of training epochs
- `batch_size`: Training batch size per device
- `learning_rate`: Learning rate (default: 5e-5)
- `warmup_steps`: Learning rate warmup steps
- `weight_decay`: Weight decay for regularization
- `gradient_accumulation_steps`: Accumulate gradients over N steps

### Evaluation Settings

- `eval_steps`: Evaluate every N steps
- `save_steps`: Save checkpoint every N steps
- `save_total_limit`: Maximum number of checkpoints to keep
- `logging_steps`: Log metrics every N steps

### Hardware Settings

- `use_fp16`: Enable mixed precision training (requires CUDA)
- `use_cuda`: Use CUDA if available
- `num_workers`: Number of data loading workers

### Optional Integrations

- `use_wandb`: Enable Weights & Biases experiment tracking
- `wandb_project`: W&B project name
- `wandb_run_name`: W&B run name (optional)

### Early Stopping

- `early_stopping`: Enable early stopping
- `early_stopping_patience`: Stop if no improvement for N evaluations
- `early_stopping_threshold`: Minimum improvement threshold

## Data Pipeline Integration

The enhanced trainer integrates seamlessly with the data pipeline:

### Step 1: Prepare Data

```bash
# Build corpus
python src/data_scraper/processing/build_unified_corpus.py \
    --strategy modern \
    --output data/pre_training/tamil_texts/all_sentences_modern.jsonl

# Export to HuggingFace format
python src/data_scraper/export/export_unified_hf.py \
    --input data/pre_training/tamil_texts/all_sentences_modern.jsonl \
    --output data/pre_training/tamil_texts/hf \
    --strategy modern
```

### Step 2: Train Model

```bash
python src/models/sangam_gpt/train_enhanced.py --config config.yaml
```

## Complete End-to-End Example

```bash
#!/bin/bash
# Complete training pipeline

# 1. Build modern Tamil corpus
echo "Building corpus..."
python src/data_scraper/processing/build_unified_corpus.py \
    --strategy modern \
    --output data/pre_training/tamil_texts/all_sentences_modern.jsonl

# 2. Export to HuggingFace format
echo "Exporting to HF format..."
python src/data_scraper/export/export_unified_hf.py \
    --input data/pre_training/tamil_texts/all_sentences_modern.jsonl \
    --output data/pre_training/tamil_texts/hf \
    --strategy modern

# 3. Create training configuration
echo "Creating config..."
python src/models/sangam_gpt/train_enhanced.py \
    --create-config training_config.yaml

# 4. Train model
echo "Training model..."
python src/models/sangam_gpt/train_enhanced.py \
    --config training_config.yaml \
    --use-wandb

echo "Training complete!"
```

## Features

### ✅ Improvements Over Original train.py

1. **Configuration Management**
   - YAML/JSON config file support
   - Command-line override
   - Save/load configurations

2. **Data Pipeline Integration**
   - Direct HuggingFace dataset loading
   - No hardcoded paths
   - Support for train/val/test splits

3. **Better Model Loading**
   - No `local_files_only` requirement
   - Automatic download fallback
   - Flexible tokenizer handling

4. **Evaluation & Metrics**
   - Automatic evaluation during training
   - Metric logging and saving
   - Early stopping support

5. **Experiment Tracking**
   - Weights & Biases integration
   - Detailed logging
   - Progress tracking

6. **Production Ready**
   - Error handling
   - Type hints
   - Comprehensive logging
   - Checkpoint management

## Legacy Script

The original `train.py` is still available but has limitations:

- Hardcoded file paths
- No configuration file support
- `local_files_only=True` breaks without cached model
- No evaluation metrics
- No experiment tracking

**Recommendation**: Use `train_enhanced.py` for all new training runs.

## Troubleshooting

### CUDA Out of Memory

Reduce batch size and/or max_length:

```yaml
batch_size: 2
max_length: 256
gradient_accumulation_steps: 2  # Effective batch size = 2 * 2 = 4
```

### Model Download Fails

Ensure you have internet connection or download model manually:

```bash
python -c "from transformers import AutoModel; AutoModel.from_pretrained('sangam/IndianLanguages-Tamil-BERT-v0.1')"
```

### Data Not Found

Verify data files exist:

```bash
ls data/pre_training/tamil_texts/hf/
# Should show: train.jsonl, validation.jsonl, test.jsonl, README.md
```

## Example Output

```
2024-01-15 10:00:00 - INFO - Configuration loaded from config.yaml
2024-01-15 10:00:01 - INFO - Using CUDA device: NVIDIA GeForce RTX 3090
2024-01-15 10:00:05 - INFO - Tokenizer loaded. Vocab size: 30000
2024-01-15 10:00:10 - INFO - Model loaded: 124,000,000 total parameters, 124,000,000 trainable
2024-01-15 10:00:15 - INFO - Dataset loaded:
2024-01-15 10:00:15 - INFO -   train: 2334 examples
2024-01-15 10:00:15 - INFO -   validation: 292 examples
2024-01-15 10:00:15 - INFO -   test: 292 examples
2024-01-15 10:00:20 - INFO - Starting training...
...
2024-01-15 12:30:45 - INFO - Training complete! Model saved to models/tamil_model
2024-01-15 12:30:45 - INFO - Final training loss: 2.345
```

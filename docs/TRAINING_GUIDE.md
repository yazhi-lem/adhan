# Adhan Model Training Guide

## Overview

This guide walks through the complete process of training the Adhan Tamil language model using the modern Tamil corpus.

**Status**: Ready for training  
**Framework**: Hugging Face Transformers  
**Model Base**: XLM-RoBERTa-base  
**Task**: Masked Language Modeling (MLM)  
**Corpus**: Modern Tamil Enhanced (1,526 records)  

---

## Training Notebooks

### 1️⃣ `01_setup_and_exploration.ipynb` - Environment Setup

**What it does**:
- Installs all dependencies (transformers, datasets, torch, etc.)
- Sets up project directories and paths
- Loads and explores the modern Tamil corpus
- Analyzes dataset statistics (source distribution, quality metrics, text lengths)
- Loads XLM-RoBERTa tokenizer and tests on Tamil text
- Tokenizes datasets and prepares for training

**Key Outputs**:
- Tokenized datasets saved to `models/tokenized_datasets/`
- Data exploration visualization: `notebooks/01_data_exploration.png`
- Dataset statistics and summaries

**Duration**: ~10-15 minutes
**GPU Required**: No
**Key Metrics**:
- Train: 1,220 records
- Val: 152 records
- Test: 154 records
- Tokenizer: xlm-roberta-base (250K vocab)

---

### 2️⃣ `02_model_training.ipynb` - MLM Pretraining

**What it does**:
- Loads pre-tokenized datasets from notebook 1
- Loads XLM-RoBERTa-base pretrained model (124M parameters)
- Configures masked language modeling (MLM) with 15% masking
- Sets up training arguments with optimized hyperparameters
- Trains the model for 10 epochs
- Evaluates on validation and test sets
- Tests with fill-mask predictions
- Saves trained model

**Training Configuration**:
```
Epochs: 10
Batch Size: 32 per device
Learning Rate: 5e-5
Warmup Steps: 100
Weight Decay: 0.01
Optimizer: AdamW
Mixed Precision: Yes (FP16)
Total Training Steps: ~380 (38 steps/epoch × 10 epochs)
```

**Key Outputs**:
- Trained model: `models/adhan-mlm-v1/`
- Checkpoints: `models/checkpoints/`
- Training logs: `logs/`
- Metrics visualization: `notebooks/02_training_metrics.png`
- Results summary: `models/adhan-mlm-v1/training_results.json`

**Duration**: 
- GPU (NVIDIA A100): ~20-30 minutes
- GPU (NVIDIA V100): ~45-60 minutes
- GPU (NVIDIA T4): ~2-3 hours
- CPU: Not recommended (>6 hours)

**Expected Results**:
- Test Loss: ~2.5-3.0
- Test Perplexity: ~12-20
- Model size: ~500 MB

---

### 3️⃣ `03_downstream_tasks.ipynb` - Fine-tuning (Optional)

**What it does** (planned):
- Loads trained Adhan model
- Fine-tunes for specific downstream tasks:
  - Named Entity Recognition (NER)
  - Sentiment Analysis
  - Text Classification
  - Question Answering
- Evaluates on task-specific benchmarks
- Provides inference examples

**Status**: To be created

---

## Quick Start

### Step 1: Run Setup Notebook

```bash
cd /home/neutron/.openclaw/zorba/Projects/OSS/yazhi/models/adhan
jupyter notebook notebooks/01_setup_and_exploration.ipynb
```

Execute all cells. This prepares the tokenized datasets.

**Expected Output**:
```
✅ Datasets created in models/tokenized_datasets/
- train.arrow (compressed tokenized data)
- val.arrow
- test.arrow
```

### Step 2: Run Training Notebook

```bash
jupyter notebook notebooks/02_model_training.ipynb
```

Execute all cells to train the model. **Ensure GPU is available** for faster training.

**Expected Output**:
```
✅ Model saved to models/adhan-mlm-v1/
- pytorch_model.bin (500 MB)
- config.json
- tokenizer.json
- training_results.json
```

### Step 3: Test the Trained Model

```python
from transformers import pipeline

fill_mask = pipeline(
    "fill-mask",
    model="models/adhan-mlm-v1",
    local_files_only=True
)

result = fill_mask("தமிழ் [MASK] என்பது மொழியாகும்।")
for pred in result[:5]:
    print(f"{pred['token_str']}: {pred['score']:.4f}")
```

---

## Hardware Requirements

### Minimum Requirements
- **CPU**: 8-core processor (Intel i7/AMD Ryzen 7)
- **RAM**: 16 GB
- **Storage**: 50 GB free (for model, datasets, checkpoints)
- **GPU**: Not required, but highly recommended

### Recommended for Training
- **GPU Memory**: 8 GB minimum
  - RTX 3080 (10 GB) ✅ Excellent
  - RTX A100 (40 GB) ✅ Ideal
  - V100 (16 GB) ✅ Good
  - T4 (16 GB) ✅ Good
  - RTX 2060 (6 GB) ⚠️ May require gradient accumulation
  
- **CPU**: Modern multi-core (2024+)
- **RAM**: 32 GB+ recommended
- **Storage**: 100 GB free

### For Inference Only
- **GPU Memory**: 2 GB minimum
- **RAM**: 8 GB
- **Storage**: 20 GB

---

## Training Hyperparameters Guide

### Current Configuration (Recommended)

```python
{
    'learning_rate': 5e-5,          # Start conservative
    'num_train_epochs': 10,         # Full pretraining
    'per_device_train_batch_size': 32,
    'per_device_eval_batch_size': 64,
    'warmup_steps': 100,            # 5% of training
    'weight_decay': 0.01,           # L2 regularization
    'mlm_probability': 0.15,        # Standard BERT masking
    'max_position_embeddings': 512, # Max sequence length
    'gradient_accumulation_steps': 1,
    'fp16': True,                   # Mixed precision
}
```

### Adjustment Guidelines

**If training is too slow**:
- Reduce `per_device_train_batch_size` → 16 (use gradient accumulation: 2)
- Reduce `num_train_epochs` → 5
- Use `gradient_accumulation_steps` to maintain effective batch size

**If training is unstable** (loss spikes):
- Reduce `learning_rate` → 2e-5
- Increase `warmup_steps` → 200
- Check `weight_decay` is correct (0.01 is standard)

**If GPU memory is full**:
- Reduce batch size → 16, 8, or 4
- Enable `fp16` (mixed precision) to save memory
- Enable gradient accumulation

**For faster convergence**:
- Increase `learning_rate` → 1e-4
- Reduce `warmup_steps` → 50
- Increase `mlm_probability` → 0.20

---

## Data Flow

```
Raw Corpus
├── Project Madurai (classical)
├── Wikipedia (formal)
├── News articles (modern)
└── Social media (colloquial)
         ↓
    [MODERN CORPUS PIPELINE]
         ↓
data/pre_training/tamil_texts/
├── all_sentences.jsonl (2,918 records)
├── all_sentences_modern.jsonl (3,066 records) ← USED
└── hf/
    ├── train.jsonl (1,220 records)
    ├── validation.jsonl (152 records)
    └── test.jsonl (154 records)
         ↓
    [Notebook 1: Tokenization]
         ↓
models/tokenized_datasets/
├── train/ (tokenized train data)
├── val/ (tokenized val data)
└── test/ (tokenized test data)
         ↓
    [Notebook 2: Training]
         ↓
models/adhan-mlm-v1/ ← TRAINED MODEL
├── pytorch_model.bin (500 MB)
├── config.json
├── tokenizer.json
└── training_results.json
```

---

## Expected Training Progress

### Training Loss Curve (typical)

```
Epoch 1:  Loss ~3.2  → 2.8
Epoch 2:  Loss ~2.6  → 2.4
Epoch 3:  Loss ~2.3  → 2.1
Epoch 4:  Loss ~2.0  → 1.9
Epoch 5:  Loss ~1.8  → 1.7
Epoch 6:  Loss ~1.7  → 1.6
Epoch 7:  Loss ~1.6  → 1.5
Epoch 8:  Loss ~1.5  → 1.4
Epoch 9:  Loss ~1.4  → 1.4
Epoch 10: Loss ~1.3  → 1.3
```

**Typical Final Metrics**:
- Train Loss: ~1.3
- Val Loss: ~2.5
- Test Loss: ~2.7
- Test Perplexity: ~14.9

_Note: Actual values may vary based on data and hardware_

---

## Monitoring Training

### Via Tensorboard

```bash
tensorboard --logdir logs/
```

Opens at `http://localhost:6006/`

### Via Console Output

The trainer prints:
- Global step count
- Current training loss
- Validation metrics (every epoch)
- Estimated time remaining

### Via Checkpoint Files

Checkpoints are saved to `models/checkpoints/` after each epoch:
```
checkpoint-38/   (epoch 1)
checkpoint-76/   (epoch 2)
checkpoint-114/  (epoch 3)
...
checkpoint-380/  (epoch 10)
```

---

## Troubleshooting

### Issue: CUDA Out of Memory

**Solution**:
```python
# Reduce batch size
per_device_train_batch_size = 16
# Enable gradient accumulation
gradient_accumulation_steps = 2
# Use mixed precision
fp16 = True
```

### Issue: Training Loss Not Decreasing

**Solution**:
- Check learning rate (try 2e-5 or 1e-4)
- Increase warmup steps (200-500)
- Verify data is correctly tokenized
- Check if dataset has adequate variety

### Issue: Very Slow Training on CPU

**Solution**:
- Use Google Colab with GPU
- Use cloud GPU services (AWS, GCP, Azure)
- Request GPU access: `torch.cuda.is_available()`

### Issue: Model Inference is Slow

**Solution**:
- Use smaller model: DistilBERT-base
- Quantize model with ONNX
- Use batch inference
- Enable GPU inference

### Issue: Tokenizer Issues with Tamil Text

**Solution**:
```python
# Verify UTF-8 encoding
text.encode('utf-8')  # should work

# If mojibake (garbled), fixing TSCII:
from src.data_scraper.raw_extractors.tamil_corpus_scraper import tscii_to_unicode
text = tscii_to_unicode(text)
```

---

## Performance Optimization

### For Training Speed

1. **Enable Mixed Precision** (save 40% memory):
   ```python
   fp16 = True  # or bf16 for newer GPUs
   ```

2. **Use Gradient Accumulation** (if GPU memory limited):
   ```python
   gradient_accumulation_steps = 2  # Effective batch size = 32×2
   ```

3. **Increase Number of Workers** (CPU data loading):
   ```python
   dataloader_num_workers = 4
   dataloader_pin_memory = True
   ```

4. **Reduce Max Sequence Length** (if text allows):
   ```python
   max_length = 256  # Default 512
   ```

### For Inference Speed

1. **Use Smaller Model**:
   - xlm-roberta-large → xlm-roberta-base (2x faster)
   - DistilBERT (3x faster than BERT)

2. **Quantization**:
   ```python
   model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
   ```

3. **Batch Inference**:
   ```python
   results = fill_mask([text1, text2, text3])  # Process multiple texts
   ```

---

## Next Steps After Training

### 1. Fine-tune for Downstream Tasks
- Named Entity Recognition (NER)
- Sentiment Analysis
- Text Classification

### 2. Evaluate on Benchmarks
- Tamil Corpus Evaluation
- Comparative analysis vs. other Tamil LLMs

### 3. Create Instruction-Tuning Dataset
- Collect conversation pairs
- Annotate for quality
- Train instruction-following model

### 4. Deploy Model
- Convert to ONNX format
- Quantize for edge devices
- Create serving API

### 5. Collect Human Feedback
- A/B testing
- Preference annotations
- Iterative improvement

---

## References & Resources

### Official Documentation
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/)
- [Trainer API](https://huggingface.co/docs/transformers/en/main_classes/trainer)
- [XLM-RoBERTa](https://huggingface.co/xlm-roberta-base)

### Tamil NLP Resources
- [Open-Tamil](https://github.com/Ezhil-Tamil/Open-Tamil)
- [Tamil Wikipedia](https://ta.wikipedia.org/)
- [Project Madurai](http://projectmadurai.org/)

### Model Cards
- Base model: [xlm-roberta-base](https://huggingface.co/xlm-roberta-base)
- Our model: See `models/adhan-mlm-v1/README.md`

---

## FAQ

**Q: Can I train on CPU?**  
A: Yes, but very slow (~6+ hours). GPU is recommended (1-3 hours).

**Q: What if I don't have GPU?**  
A: Use Google Colab (free), AWS, GCP, or rent GPU time.

**Q: Can I use smaller model?**  
A: Yes, DistilBERT is 3x faster and similar quality.

**Q: How long does training take?**  
A: 1-3 hours on modern GPU (RTX 3080+).

**Q: Can I resume training from checkpoint?**  
A: Yes, set `resume_from_checkpoint=True` in trainer.

**Q: How do I use the model for inference?**  
A: See `02_model_training.ipynb` section 9 for examples.

---

**Last Updated**: February 18, 2026  
**Status**: Ready for training  
**Next**: Execute notebooks in order

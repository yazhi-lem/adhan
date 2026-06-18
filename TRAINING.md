# ADHAN TRAINING GUIDE

**Model:** Adhan — Tamil LLM (Decoder-Only Transformer)  
**Script:** `models/adhan/scripts/train_adhan_real.py`  
**Updated:** 2026-06-18 | Rotation 26 Cycle 2

---

## Quick Start

```bash
# 1. Verify PyTorch is installed:
python3 -c "import torch; print(torch.__version__)"

# 2. Run smoke test (no data needed):
python3 models/adhan/scripts/train_adhan_real.py --smoke_test

# 3. Train on OpenSangam data:
python3 models/adhan/scripts/train_adhan_real.py \
  --train_data models/sangam/release/v1.0.0/data/train.jsonl \
  --val_data models/sangam/release/v1.0.0/data/val.jsonl \
  --tokenizer models/yazh/yazh-tokenizer.json \
  --output models/adhan/checkpoints/real-v1 \
  --epochs 10 --batch_size 16 --lr 3e-4
```

---

## Architecture

```
AdhanTransformer
├── Token Embedding (65K x 512)
├── x8 Transformer Blocks:
│   ├── Pre-Norm + RoPE Multi-Head Attention (8 heads, 64-dim)
│   ├── Pre-Norm + SwiGLU Feedforward (512 -> 2048 -> 512)
│   └── Residual connections
├── LayerNorm
└── LM Head (tied with embedding) -> 65K logits
```

**Parameters:** ~85M  
**Context:** 512 tokens  
**Optimizer:** AdamW (lr=3e-4, weight_decay=0.01)  
**Scheduler:** Cosine with linear warmup (100 steps)  
**Precision:** AMP (fp16 on GPU), fp32 on CPU

### Tamil-Specific Design Choices

| Choice | Rationale |
|--------|-----------|
| **RoPE** (not learned pos emb) | Better generalization for agglutinative morphology; handles variable-length inputs at inference |
| **SwiGLU** (not ReLU) | Better gradient flow for deep models; proven in PaLM/Llama |
| **Pre-norm** (not post-norm) | More stable training for 8+ layers |
| **512 context** (not 256) | Tamil words are longer due to agglutination; need more tokens per sentence |
| **65K vocab** (not 30K-50K) | Covers full Tamil Unicode range + compound forms |
| **Weight tying** | Reduces parameters by ~15M; improves perplexity |

---

## Hardware Requirements

### Minimum (CPU Training)

- **Device:** RPi 5 (8GB RAM) or any x86_64 machine
- **RAM:** 4GB minimum (8GB recommended)
- **Disk:** 5GB for checkpoints + data
- **PyTorch:** CPU version (pip install torch)
- **Settings:**
  ```
  --batch_size 2 --grad_accum 8 --n_layers 4 --d_model 256
  ```
- **Expected time:** ~2-4 hours per epoch on RPi 5 (OpenSangam 5,809 entries)
- **Notes:** Use `n_layers 4, d_model 256` variant for RPi (~20M params)

### Recommended (GPU Training)

- **Device:** NVIDIA GPU with 8GB+ VRAM (RTX 3070 / A4000)
- **RAM:** 16GB system RAM
- **Disk:** 20GB (SSD recommended)
- **PyTorch:** CUDA version (torch+cuda)
- **Settings:**
  ```
  --batch_size 16 --grad_accum 1 --d_model 512 --n_layers 8
  ```
- **Expected time:** ~5-15 minutes per epoch (full model, 85M params)
- **Expected total:** 1-3 hours for 10 epochs

### Ideal (Multi-GPU Training)

- **Device:** 2-4x NVIDIA A100/H100 (or S-Node cluster)
- **Settings:**
  ```
  --batch_size 32 --grad_accum 1 --d_model 768 --n_layers 12
  ```
- **Expected time:** ~10-30 minutes for full training
- **Notes:** Use `torchrun` for distributed training (coming in future update)

---

## Training by Phase

### Phase 1: OpenSangam Classical (Available Now)

```bash
python3 train_adhan_real.py \
  --train_data models/sangam/release/v1.0.0/data/train.jsonl \
  --val_data models/sangam/release/v1.0.0/data/val.jsonl \
  --tokenizer models/yazh/release/v1.0.0/yazh-tokenizer.json \
  --output models/adhan/checkpoints/phase1-classical \
  --epochs 20 --batch_size 16 --lr 3e-4 --warmup_steps 200
```

- **Data:** 5,809 train / 726 val entries
- **Expected perplexity:** 15-30 (after 20 epochs)
- **Time:** ~30 min (GPU) / ~8 hours (RPi 5)

### Phase 2: News Tamil (Scraper Required)

```bash
python3 train_adhan_real.py \
  --train_data models/adhan/data/news-train.jsonl \
  --val_data models/adhan/data/news-val.jsonl \
  --tokenizer models/yazh/release/v1.0.0/yazh-tokenizer.json \
  --output models/adhan/checkpoints/phase2-news \
  --epochs 10 --batch_size 16 --lr 2e-4 --resume models/adhan/checkpoints/phase1-classical/checkpoint-best.pt
```

- **Data:** ~10,000+ news articles (news scraper output)
- **Skip:** Continue training from Phase 1 checkpoint
- **Expected time:** ~1 hour (GPU)

### Phase 3: Colloquial Tamil (Corpus Required)

```bash
python3 train_adhan_real.py \
  --train_data models/adhan/data/colloquial-train.jsonl \
  --val_data models/adhan/data/colloquial-val.jsonl \
  --tokenizer models/yazh/release/v1.0.0/yazh-tokenizer.json \
  --output models/adhan/checkpoints/phase3-colloquial \
  --epochs 10 --batch_size 16 --lr 2e-4
```

- **Data:** 10K+ colloquial Tamil sentences (podcasts, dialogues)
- **Expected time:** ~1 hour (GPU)

### Phase 4: Combined Corpus (All Sources)

```bash
python3 train_adhan_real.py \
  --train_data models/adhan/data/combined-train.jsonl \
  --val_data models/adhan/data/combined-val.jsonl \
  --tokenizer models/yazh/release/v1.0.0/yazh-tokenizer.json \
  --output models/adhan/checkpoints/phase4-combined \
  --epochs 10 --batch_size 32 --lr 3e-4 --warmup_steps 500
```

- **Data:** All sources combined (~100K+ entries target)
- **Preprocessing:** Concatenate all jsonl files, re-shuffle, re-split (80/10/10)

---

## Checkpointing

Checkpoints are saved to the `--output` directory:

```
models/adhan/checkpoints/real-v1/
├── checkpoint-latest.pt       # Most recent (always overwritten)
├── checkpoint-best.pt         # Lowest validation loss
├── checkpoint-step-500.pt     # Periodic (every --ckpt_every steps)
├── checkpoint-step-1000.pt
└── ...
```

Each checkpoint contains:
- Model weights
- Optimizer state (for resuming)
- Scheduler state
- Training step + epoch
- Validation metrics

### Resume Training

```bash
python3 train_adhan_real.py \
  --resume models/adhan/checkpoints/real-v1/checkpoint-best.pt \
  --train_data ... --val_data ... --output ...
```

---

## RPi 5 Configuration Guide

Zorba (RPi 5, 8GB RAM, armhf userspace):

```bash
# Use small model variant
python3 train_adhan_real.py \
  --train_data models/sangam/release/v1.0.0/data/train.jsonl \
  --val_data models/sangam/release/v1.0.0/data/val.jsonl \
  --output models/adhan/checkpoints/rpi-test \
  --epochs 5 --batch_size 2 --grad_accum 8 \
  --n_layers 4 --d_model 256 --d_ff 1024 \
  --lr 1e-4 --warmup_steps 50
```

**RPi 5 Expected Performance:**

| Config | Params | Memory | Time/Epoch |
|--------|--------|--------|------------|
| Tiny (2L, 64D) | ~1M | 500MB | ~10 min |
| Small (4L, 256D) | ~20M | 2.5GB | ~45 min |
| Medium (6L, 384D) | ~50M | 4.5GB | ~2 hours |
| Full (8L, 512D) | ~85M | 7.5GB | ~4 hours (may OOM) |

**Recommendations for RPi 5:**
- Use Small config (4L, 256D) — fits in RAM, reasonable speed
- Close all other applications (RPi 5 8GB is tight for full model)
- Increase SWAP to 2GB if training full model: `sudo dphys-swapfile swapoff && sudo nano /etc/dphys-swapfile`
- Disable desktop: `sudo systemctl set-default multi-user.target`
- Monitor temp: `vcgencmd measure_temp` (keep < 80C)

---

## Evaluating Results

### Metrics to Watch

| Metric | Good | Concern | Bad |
|--------|------|---------|-----|
| Train loss (10 epochs) | < 1.5 | 1.5-3.0 | > 3.0 |
| Val loss (10 epochs) | < 2.0 | 2.0-4.0 | > 4.0 |
| Perplexity | < 10 | 10-50 | > 50 |
| Overfitting gap | < 0.5 | 0.5-1.0 | > 1.0 |

### If Training Loss Doesn't Decrease
- Increase learning rate: `--lr 5e-4`
- Check data encoding: tokenizer may be splitting Tamil incorrectly
- Reduce model size: fewer layers or smaller d_model
- Increase warmup: `--warmup_steps 500`

### If Validation Loss >> Train Loss (Overfitting)
- Increase dropout: `--dropout 0.2`
- Reduce model size
- Add more training data
- Use weight decay (default: 0.01)

### If Training is Too Slow
- Reduce `--max_seq_len` to 256
- Use gradient accumulation: `--batch_size 2 --grad_accum 8`
- Reduce model size: `--n_layers 4 --d_model 256`
- Close other applications (free RAM)

---

## Dependency Installation

### PyTorch (CPU)
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### PyTorch (CUDA 12.1)
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### Verify
```bash
python3 -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

---

## Data Format

Training data must be JSONL (one JSON object per line):

```jsonl
{"text": "அகர முதல எழுத்தெல்லாம் ஆதி பகவன் முதற்றே உலகு", "source": "thirukkural", "author": "Thiruvalluvar"}
{"text": "பக்கத்து வீடுகளிருந்து ஒவ்வொருவராக ஓடிவர ஆரம்பித்தார்கள்", "source": "tamil_stories"}
```

The script reads the `text` field. Other fields are ignored.

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError: torch` | Install PyTorch (see above) |
| `CUDA out of memory` | Reduce batch_size, use grad_accum, reduce model size |
| `Killed` (OOM on RPi) | Use smaller model, increase SWAP, close other apps |
| Loss is NaN | Reduce learning rate, check for bad data, increase warmup |
| Loss doesn't decrease | Check tokenizer, increase LR, verify data format |
| Slow data loading | Reduce num_workers, use SSD, check disk I/O |

---

## Next Steps After Training

1. **Evaluate:** Run validation on test set
2. **Generate:** Use `generate.py` (coming soon) for text generation
3. **Export:** Convert to ONNX for mobile deployment
4. **Quantize:** INT8/INT4 quantization for on-device inference
5. **Upload:** Push to Hugging Face Hub

---

*Reference: docs/TAMIL_FIRST_DOCTRINE.md | memory/2026-06-18-STRATEGIC-PIVOT-LAYER4.md*  
*Co-Authored-By: Hermes <hermes@yazhi.org>*

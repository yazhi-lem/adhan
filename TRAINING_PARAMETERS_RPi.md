# Tamil Model Training Parameters — RPi Edge Deployment Guide
**ARIVU | Rotation 26 Cycle 5+ | Jul 1, 2026**

---

## OVERVIEW

This document specifies training hyperparameters tuned for Adhan on Tamil linguistic data, optimized for deployment on Raspberry Pi 5 edge clusters (4× RPi 5 = 16 GB RAM, 16 CPU cores shared).

**Design Philosophy:**
- Gradient accumulation to simulate large effective batch sizes
- Memory-efficient tokenization + streaming data
- Learning rate schedules validated for Tamil morphological stability
- Checkpoint strategies optimized for RPi storage constraints

---

## HARDWARE TARGETS

### Development Machine (GPU)
- Device: Any modern GPU (RTX 3090 / A100)
- Memory: 16–40 GB
- Training: Single device, batch_size=16, no gradient accumulation

### Raspberry Pi 5 (Single Node)
- RAM: 8 GB
- Storage: 256 GB microSD (bottleneck)
- CPU: ARM64 (4-core @ 2.4 GHz)
- PyTorch Build: CPU-only, optimized for ARM
- Effective Batch: batch_size=2 × grad_accum=8 = 16

### RPi Cluster (4× RPi 5 = Multi-Node)
- Total RAM: 32 GB shared
- Total Storage: 1 TB distributed RAID
- Distributed strategy: Data parallelism (each node: batch=2, grad_accum=4)
- Coordination: NFS mount for shared weights

---

## TRAINING CONFIGURATIONS

### Configuration 1: Quick Test (10 min, ~5K steps)
**Target:** Smoke test, validation pipeline
**Hardware:** Any device

```yaml
model_config:
  vocab_size: 65000
  d_model: 256                    # Tiny model
  n_heads: 4                      # Single head per layer
  n_layers: 4
  d_ff: 512                       # Smaller intermediate
  max_seq_len: 128
  dropout: 0.1
  
  # Parameters: ~10M (vs 85M full)

training_config:
  batch_size: 8
  grad_accum: 1
  epochs: 1
  max_steps: 500                  # ~10 minutes of training
  
  lr_schedule: "cosine_warmup"
  learning_rate: 5e-4             # Higher for tiny model
  warmup_ratio: 0.1               # 50 steps warmup
  min_lr: 1e-6
  
  optimizer: "adamw"
  beta1: 0.9
  beta2: 0.95
  eps: 1e-8
  weight_decay: 0.01
  
  grad_clip_norm: 1.0
  
  early_stop_patience: 3
  save_interval: 100              # Save every 100 steps
  
  data:
    train_data: "test_scrape.jsonl"
    val_data: "test_scrape.jsonl"
    shuffle: true
    shuffle_buffer_size: 1000
```

**Expected Performance:**
- Loss: ~3.0 → ~2.0 across 500 steps
- Time: 8-12 minutes on GPU, 40-60 min on RPi single-core
- Memory: ~300 MB peak

**Command:**
```bash
python3 train_adhan_real.py --smoke_test
```

---

### Configuration 2: Single RPi 5 (Local, Offline)
**Target:** Adaptive learning, local corpus (synthetic data)
**Hardware:** Raspberry Pi 5 (8 GB RAM)
**Duration:** ~2 hours for 1 epoch

```yaml
model_config:
  vocab_size: 65000
  d_model: 384                    # Smaller than 512 (RPi constraints)
  n_heads: 6                      # 384 / 6 = 64 dim/head
  n_layers: 6                     # 6 layers instead of 8
  d_ff: 1024
  max_seq_len: 256                # Shorter sequences to fit memory
  dropout: 0.1
  
  # Parameters: ~45M (half-size for RPi)

training_config:
  batch_size: 2                   # Very small for RPi memory
  grad_accum: 8                   # 2 × 8 = 16 effective batch → gradient stability
  
  epochs: 5
  max_steps: null                 # Run full epochs
  
  lr_schedule: "cosine_warmup"
  learning_rate: 1.5e-4           # LOWER for small device (more conservative)
  warmup_steps: 200               # Longer warmup for stability
  min_lr: 5e-7
  
  optimizer: "adamw"
  beta1: 0.9
  beta2: 0.95
  eps: 1e-8
  weight_decay: 0.01              # Regularization important on limited data
  
  grad_clip_norm: 0.5             # More aggressive clipping (conservative)
  
  early_stop_patience: 2
  save_interval: 50               # Save every 50 steps
  keep_n_checkpoints: 3           # Storage-constrained
  
  data:
    train_data: "synthetic_tamil_train.jsonl"
    val_data: "synthetic_tamil_val.jsonl"
    shuffle: true
    shuffle_buffer_size: 500       # Smaller buffer (memory)
```

**Expected Performance:**
- Loss: ~2.8 → ~2.2 over 5 epochs
- Time: ~2 hours per epoch on RPi 5 (ARM CPU)
- Memory: ~450 MB peak (streaming)
- Storage: ~500 MB for checkpoints

**Environment:**
```bash
# Install PyTorch CPU-optimized for ARM64
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu

# Run training
python3 train_adhan_real.py \
  --batch_size 2 \
  --grad_accum 8 \
  --epochs 5 \
  --lr 1.5e-4 \
  --warmup_steps 200 \
  --output ~/adhan_checkpoints_local
```

---

### Configuration 3: Full Training (GPU or Server)
**Target:** Production model (85M parameters)
**Hardware:** GPU or multi-core CPU
**Duration:** ~8 hours for 10 epochs

```yaml
model_config:
  vocab_size: 65000
  d_model: 512                    # Full size
  n_heads: 8
  n_layers: 8
  d_ff: 2048                      # SwiGLU intermediate
  max_seq_len: 512
  dropout: 0.1
  rope: true                      # Rotary embeddings
  
  # Parameters: ~85M

training_config:
  batch_size: 16                  # Full batch for GPU
  grad_accum: 1
  epochs: 10
  max_steps: null
  
  lr_schedule: "cosine_warmup"
  learning_rate: 3.0e-4           # Standard for transformers
  warmup_ratio: 0.05              # 5% warmup (conservative)
  min_lr: 1e-6
  
  optimizer: "adamw"
  beta1: 0.9
  beta2: 0.95
  eps: 1e-8
  weight_decay: 0.01
  
  grad_clip_norm: 1.0
  
  early_stop_patience: 5
  save_interval: 100
  keep_n_checkpoints: 5
  
  mixed_precision: "fp16"         # GPU: 50% memory reduction
  
  data:
    train_data: "opensangam_classical_train.jsonl"
    val_data: "opensangam_classical_val.jsonl"
    # Multi-source (when available):
    # - news_tamil_train.jsonl
    # - colloquial_tamil_train.jsonl
    shuffle: true
    shuffle_buffer_size: 10000
```

**Expected Performance:**
- Loss: ~2.5 → ~1.8 over 10 epochs
- Time: ~8 hours on RTX 3090, ~12 hours on A100
- Memory: ~16 GB (FP32) / ~8 GB (FP16)
- Throughput: ~500 tokens/sec (GPU)

**Command:**
```bash
python3 train_adhan_real.py \
  --batch_size 16 \
  --epochs 10 \
  --lr 3e-4 \
  --warmup_ratio 0.05 \
  --mixed_precision fp16 \
  --output ~/adhan_checkpoints_prod
```

---

### Configuration 4: RPi 4-Node Cluster (Distributed)
**Target:** Parallel training across 4 RPi 5 nodes
**Hardware:** 4× RPi 5 (32 GB total RAM)
**Throughput:** ~250 tokens/sec (4 nodes × 62 tokens/sec/node)

```yaml
model_config:
  vocab_size: 65000
  d_model: 384                    # Smaller model for RPi cluster
  n_heads: 6
  n_layers: 6
  d_ff: 1024
  max_seq_len: 256
  dropout: 0.1
  
  # Parameters: ~45M per replica

distributed_config:
  backend: "nccl"                 # CPU: gloo, GPU: nccl
  world_size: 4                   # 4 nodes
  rank: 0                         # Node index (0-3)
  master_addr: "192.168.1.50"    # RPi 0 (primary)
  master_port: 29500
  
training_config:
  batch_size: 2                   # Per-node batch
  grad_accum: 4                   # Per-node accumulation
  # Global effective batch = 4 nodes × 2 batch × 4 accum = 32
  
  epochs: 5
  
  lr_schedule: "cosine_warmup"
  learning_rate: 2.0e-4           # Slightly higher for larger effective batch
  warmup_steps: 200
  min_lr: 5e-7
  
  optimizer: "adamw"
  beta1: 0.9
  beta2: 0.95
  eps: 1e-8
  weight_decay: 0.01
  
  grad_sync_interval: 2           # Sync gradients every 2 accumulation steps
  
  data:
    train_data: "synthetic_tamil_train_distributed.jsonl"
    val_data: "synthetic_tamil_val.jsonl"
    # Each node samples different slices for data parallelism
    shard_by_node: true
    shuffle: true
    shuffle_buffer_size: 300
```

**Setup (on each RPi):**
```bash
# Install distributed PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Set environment
export MASTER_ADDR=192.168.1.50
export MASTER_PORT=29500
export WORLD_SIZE=4
export RANK=0  # Change per node (0, 1, 2, 3)
export LOCAL_RANK=0

# Run distributed training on Node 0
torchrun \
  --nproc_per_node=1 \
  --nnodes=4 \
  --node_rank=0 \
  --master_addr=192.168.1.50 \
  --master_port=29500 \
  train_adhan_real.py \
  --batch_size 2 \
  --grad_accum 4 \
  --epochs 5 \
  --lr 2e-4
```

**Expected Performance:**
- Throughput: ~60 tokens/sec per RPi, ~240 tokens/sec total
- Time: ~4 hours per epoch (vs 2h single RPi = 2× speedup)
- Memory: ~400 MB per node
- Network I/O: Minor (only gradient sync)

---

## LEARNING RATE SCHEDULES

### Schedule 1: Cosine Decay with Warmup
**Recommended for Tamil (most stable)**

```python
def cosine_schedule(step, total_steps, warmup_steps, lr_max=3e-4, lr_min=1e-6):
    """
    Cosine decay with linear warmup.
    warmup: 0 → lr_max over warmup_steps
    decay: lr_max → lr_min over remaining steps
    """
    if step < warmup_steps:
        return lr_max * (step / warmup_steps)
    else:
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * progress))
```

**Parameters (by config):**
- Quick Test: warmup_steps=50, lr_max=5e-4, lr_min=1e-6
- RPi Local: warmup_steps=200, lr_max=1.5e-4, lr_min=5e-7
- Full Training: warmup_steps=500, lr_max=3e-4, lr_min=1e-6
- RPi Cluster: warmup_steps=200, lr_max=2e-4, lr_min=5e-7

---

## OPTIMIZER SETTINGS

### AdamW (Recommended)
```python
optimizer_config = {
    'lr': <from schedule>,
    'betas': (0.9, 0.95),      # Standard for LLM
    'eps': 1e-8,
    'weight_decay': 0.01,      # L2 regularization
}
```

**Gradient Clipping:**
- GPU training: max_norm=1.0 (standard)
- RPi training: max_norm=0.5 (conservative, more stability)

---

## BATCH SIZE & GRADIENT ACCUMULATION

### Memory vs. Stability Trade-off

| Config | Batch | Grad Accum | Eff. Batch | Memory | Gradient Variance | Notes |
|--------|-------|-----------|-----------|--------|-------------------|-------|
| Quick Test | 8 | 1 | 8 | 250 MB | High | Fast iteration |
| RPi Single | 2 | 8 | 16 | 450 MB | Low | Stable training |
| Full GPU | 16 | 1 | 16 | 800 MB | Medium | Standard |
| RPi Cluster | 2 | 4 | 32 | 400 MB/node | Low | Multi-node sync |

**Formula for RPi:**
```
Effective Batch = batch_size × grad_accum
Memory ≈ batch_size × sequence_length × hidden_dim × precision_bytes
```

For RPi with batch=2, seq=256, d=384, FP32:
- Memory ≈ 2 × 256 × 384 × 4 = 786 KB per sequence ≈ 400 MB total

---

## DATA PIPELINE CONFIGURATION

### Classical Tamil (Recommended First)
```yaml
data:
  source: "opensangam_v1.0"
  train_split: 0.85               # 80% train, rest val
  sequences:
    - path: "opensangam_classic_train.jsonl"
      domain: "classical"
      weight: 1.0
  preprocessing:
    tokenizer: "tamil_agglutinative"
    max_seq_len: 256
    padding: "max_length"
    truncation: "longest_first"
    normalize_colloquial: false    # Keep classical pure
```

### Multi-Source (When Available)
```yaml
data:
  sequences:
    - path: "opensangam_classic_train.jsonl"
      domain: "classical"
      weight: 1.0                  # 50% of data
    - path: "scrape_news_tamil_train.jsonl"
      domain: "news"
      weight: 1.0                  # 30% of data
    - path: "colloquial_tamil_train.jsonl"
      domain: "colloquial"
      weight: 0.7                  # 20% of data (less critical)
  sampling:
    strategy: "mixed"              # Sample proportionally
    shuffle_buffer_size: 10000
```

---

## CHECKPOINTING STRATEGY

### RPi Local (Storage Constrained)
```bash
# Keep only best 3 checkpoints (save ~500 MB storage)
--keep_n_checkpoints 3
--save_interval 50

# Resume from best checkpoint
python3 train_adhan_real.py \
  --resume ~/adhan_checkpoints_local/checkpoint-best.pt \
  --epochs 10
```

### Full Training (GPU)
```bash
# Keep last 5 checkpoints
--keep_n_checkpoints 5
--save_interval 100

# Periodic snapshots every 500 steps
--snapshot_interval 500
```

---

## TAMIL-SPECIFIC TUNING NOTES

### 1. Morphological Stability
Tamil agglutination requires careful gradient updates to preserve morpheme boundaries.

**Recommendation:** Use gradient accumulation even on GPU (batch=16, accum=2) to simulate larger effective batch (32) for more stable morpheme learning.

### 2. Colloquial Normalization
Colloquial speech includes:
- Phonological reductions: போறேன் → போகிறேன்
- Grammatical simplifications: சாப்ட → சாப்பிட்ட

**Recommendation:** Train first on classical (pure morphology), then fine-tune on news + colloquial with lower learning rate (1e-4 instead of 3e-4).

### 3. Sandhi Awareness
Sound changes at morpheme boundaries (makara erukkum, nasal assimilation).

**Recommendation:** Use `tamil_agglutinative_tokenizer.py` which marks sandhi patterns as special tokens during preprocessing. Model learns these patterns implicitly.

### 4. Vocabulary Coverage
65K vocab tokens chosen to:
- Cover 99%+ of Tamil training data
- Reserve slots for morphology markers
- Support rare Grantha/Sanskrit borrowings

**Monitoring:** Track OOV rate per epoch (should be <0.5%).

---

## MONITORING & DEBUGGING

### Key Metrics to Track
```
Training Step | Loss | Val Loss | Perplexity | OOV Rate | LR | Memory
1             | 5.2  | 5.1      | 180        | 2.1%     | 5e-4 | 350MB
100           | 3.5  | 3.4      | 30         | 1.8%     | 5e-4 | 380MB
500           | 2.6  | 2.7      |11          | 1.2%     | 2e-4 | 390MB
```

### Red Flags
- **Loss not decreasing:** Learning rate too high, reduce by 50%
- **Loss plateaus:** Early stopping triggered, increase patience or use lower min_lr
- **Out of memory:** Reduce batch_size or grad_accum by 2×
- **Perplexity > 50 after 500 steps:** Data issue, check tokenizer

### Debug Mode (Single Step)
```bash
python3 train_adhan_real.py \
  --smoke_test \
  --debug \
  --batch_size 1 \
  --max_steps 1
```

---

## TIMELINE: TRAINING ON RPi 5 CLUSTER

| Phase | Duration | Config | Data | Checkpoints |
|-------|----------|--------|------|------------|
| **Phase 0: Setup & Validation** | 20 min | Smoke test | synthetic | - |
| **Phase 1: Classical Baseline** | 4h | RPi Single, 5 epochs | OpenSangam | ckpt-best |
| **Phase 2: Multi-Source** | 6h | RPi Single, 5 epochs | Classical + News | ckpt-best |
| **Phase 3: Distributed Fine-tune** | 4h | 4-node RPi cluster, 2 epochs | Full corpus | ckpt-best |
| **Phase 4: Evaluation** | 30 min | Eval framework | Test set | ✓ METRICS |

**Total:** ~15 hours wall-clock (4 phases)

---

## RECOMMENDATIONS

### For Quick Prototyping (< 1 hour)
Use **Configuration 1** (Quick Test) with synthetic data:
```bash
python3 train_adhan_real.py --smoke_test --max_steps 500
```

### For RPi Local Training (2–8 hours)
Use **Configuration 2** (Single RPi):
```bash
python3 train_adhan_real.py --batch_size 2 --grad_accum 8 --epochs 5 --lr 1.5e-4
```

### For Production (Multi-GPU or Cluster)
Use **Configuration 3** (Full Training) or **Configuration 4** (Distributed):
```bash
# GPU
python3 train_adhan_real.py --batch_size 16 --epochs 10 --lr 3e-4 --mixed_precision fp16

# RPi Cluster
torchrun --nnodes=4 --node_rank=0 train_adhan_real.py --batch_size 2 --grad_accum 4
```

---

## REFERENCES

- Asher & Kumari (1997) — *The Descriptive Grammar of Tamil*
- Varatharajan et al. (2004) — Tamil Morphology and Tokenization
- docs/TAMIL_FIRST_DOCTRINE.md — Yazhi Tamil principles
- scripts/tamil_agglutinative_tokenizer.py — Morphology-aware tokenizer
- scripts/tamil_eval_framework.py — Evaluation metrics
- scripts/synthetic_tamil_data_pipeline.py — Local data generation

---

**Last Updated:** Jul 1, 2026 | Rotation 26 Cycle 5+

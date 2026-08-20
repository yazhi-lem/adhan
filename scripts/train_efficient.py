#!/usr/bin/env python3
"""Adhan SLM — High-Efficiency Training Engine (PyTorch backend).

Second trainer alongside `adhan_slm.training.train_jax` (JAX/Flax, the primary,
documented path — see docs/CPU_TRAINING.md). This one is useful once a CUDA GPU is
available for PyTorch-native workflows (AMP, torch.compile).

Hardware is auto-detected at run time via `configure_hardware()` — CPU (any core count)
or CUDA GPU (any NVIDIA card), reported honestly from what `torch` actually finds, not
assumed from a fixed profile:
  - CPU: micro-batching + gradient accumulation, thread-pinned, bf16/fp32.
  - CUDA GPU: micro-batching + gradient accumulation, fp16 (pre-Ampere) or bf16
    (Ampere+) autocast, TF32 matmul.
  - Memory-mapped binary shard streaming (np.memmap) with zero RAM pressure.

Usage:
    # 1. Train on GPU or CPU (auto-detected):
    python scripts/train_efficient.py \
        --data-dir data/final/tamil_slm \
        --size nano \
        --batch-size 8 \
        --grad-accum 16 \
        --max-steps 10000 \
        --lr 1.5e-3

    # 2. Run overfit sanity check:
    python scripts/train_efficient.py --data-dir data/final/tamil_slm --overfit-batch

    # 3. Export trained weights directly to ONNX for Yazh Unity:
    python scripts/train_efficient.py --export-onnx --checkpoint checkpoints/adhan_nano_best.pt
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from adhan_slm.core.logging import get_logger

logger = get_logger(__name__)


# --------------------------------------------------------------------------- #
# 1. Architecture: RoPE + RMSNorm + SwiGLU + Fused Attention                  #
# --------------------------------------------------------------------------- #

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE) precomputed for fast broadcasting."""
    def __init__(self, dim: int, max_seq_len: int = 2048, theta: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        self.register_buffer("cos", freqs.cos(), persistent=False)
        self.register_buffer("sin", freqs.sin(), persistent=False)

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # q, k: [B, H, T, D]
        seq_len = q.shape[2]
        cos = self.cos[:seq_len, :].unsqueeze(0).unsqueeze(0)  # [1, 1, T, D/2]
        sin = self.sin[:seq_len, :].unsqueeze(0).unsqueeze(0)
        
        def rotate_half(x):
            x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
            return torch.cat((-x2, x1), dim=-1)
        
        q_rot = (q * torch.cat([cos, cos], dim=-1)) + (rotate_half(q) * torch.cat([sin, sin], dim=-1))
        k_rot = (k * torch.cat([cos, cos], dim=-1)) + (rotate_half(k) * torch.cat([sin, sin], dim=-1))
        return q_rot, k_rot


class Block(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.norm1 = RMSNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        
        self.norm2 = RMSNorm(d_model)
        self.gate = nn.Linear(d_model, d_ff, bias=False)
        self.up = nn.Linear(d_model, d_ff, bias=False)
        self.down = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor, rope: RotaryEmbedding) -> torch.Tensor:
        B, T, C = x.shape
        h = self.norm1(x)
        qkv = self.qkv(h).reshape(B, T, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, H, T, Dh]
        q, k = rope(q, k)
        
        # PyTorch Native Scaled Dot Product Attention (FlashAttention / Memory-Efficient)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).reshape(B, T, C)
        x = x + self.proj(out)
        
        # SwiGLU MLP
        h2 = self.norm2(x)
        x = x + self.down(F.silu(self.gate(h2)) * self.up(h2))
        return x


class AdhanModel(nn.Module):
    """Adhan Causal SLM with tied input-output embeddings."""
    def __init__(
        self,
        vocab_size: int = 12000,
        d_model: int = 256,
        n_layers: int = 6,
        n_heads: int = 4,
        d_ff: int = 1024,
        max_seq_len: int = 512
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.rope = RotaryEmbedding(d_model // n_heads, max_seq_len=max_seq_len)
        self.blocks = nn.ModuleList([Block(d_model, n_heads, d_ff) for _ in range(n_layers)])
        self.norm_f = RMSNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Weight tying (crucial for small parameter models)
        self.head.weight = self.tok_emb.weight
        
        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        x = self.tok_emb(idx)
        for block in self.blocks:
            x = block(x, self.rope)
        x = self.norm_f(x)
        return self.head(x)

    def approx_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# --------------------------------------------------------------------------- #
# 2. Fast Memory-Mapped Binary Dataset                                        #
# --------------------------------------------------------------------------- #

class MemmapShardDataset:
    """Streams fixed-length sequences from disk via np.memmap with zero memory overhead."""
    def __init__(self, bin_path: Path, seq_len: int = 512):
        self.bin_path = bin_path
        self.seq_len = seq_len
        if not bin_path.exists():
            raise FileNotFoundError(f"Shard not found: {bin_path}")
        
        # File size determines number of sequences
        file_size_bytes = bin_path.stat().st_size
        # Shards are packed as uint16 (or int32 if vocab > 65k)
        self.dtype = np.uint16
        bytes_per_tok = np.dtype(self.dtype).itemsize
        total_tokens = file_size_bytes // bytes_per_tok
        self.n_seqs = total_tokens // (seq_len + 1)
        self.mmap = np.memmap(bin_path, dtype=self.dtype, mode='r', shape=(self.n_seqs, seq_len + 1))

    def __len__(self) -> int:
        return self.n_seqs

    def get_batch(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        indices = np.random.randint(0, self.n_seqs, size=batch_size)
        rows = self.mmap[indices].astype(np.int64)
        tensor = torch.from_numpy(rows).to(device, non_blocking=True)
        return tensor[:, :-1], tensor[:, 1:]  # (input_ids, targets)


# --------------------------------------------------------------------------- #
# 3. Hardware Optimization & Device Setup                                     #
# --------------------------------------------------------------------------- #

def configure_hardware() -> Tuple[torch.device, str]:
    """Auto-detects the real CUDA GPU (if any) vs CPU and configures execution.

    Reports whatever hardware `torch` actually finds on this machine — never assumes
    a fixed GPU/CPU model, since the script is expected to run unmodified before and
    after a GPU is added to the box.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"🎮 GPU detected: {gpu_name} ({vram_gb:.1f} GB VRAM)")

        # Enable Tensor Core speedups
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        # Choose precision (fp16 for pre-Ampere, bf16 for Ampere+ where it's native)
        compute_dtype = "float16" if torch.cuda.get_device_capability()[0] < 8 else "bfloat16"
    else:
        device = torch.device("cpu")
        num_threads = os.cpu_count() or 1
        torch.set_num_threads(num_threads)
        logger.info(f"💻 CPU mode: {num_threads} threads")
        compute_dtype = "bfloat16" if hasattr(torch, "bfloat16") else "float32"

    return device, compute_dtype


# --------------------------------------------------------------------------- #
# 4. Training Engine                                                          #
# --------------------------------------------------------------------------- #

def resolve_vocab_size(data_dir: Path, cli_vocab_size: int) -> int:
    """Reads the frozen vocab size from data_dir/vocab.json when present.

    Mirrors adhan_slm.training.train_jax's behavior: the packed shards' token ids are
    only valid against the vocab they were frozen with, so trusting a CLI default here
    risks a silent embedding/shard-id mismatch if the corpus was repacked at a
    different vocab size. Falls back to the CLI value (e.g. for --overfit-batch, which
    has no shards on disk).
    """
    vocab_path = data_dir / "vocab.json"
    if not vocab_path.exists():
        return cli_vocab_size
    vocab = json.loads(vocab_path.read_text(encoding="utf-8"))
    frozen_size = len(vocab)
    if frozen_size != cli_vocab_size:
        logger.info(f"vocab_size: overriding CLI value {cli_vocab_size} with frozen {frozen_size} from {vocab_path}")
    return frozen_size


def train(args):
    device, compute_dtype = configure_hardware()
    data_dir = Path(args.data_dir)
    train_bin = data_dir / "train.bin"
    val_bin = data_dir / "val.bin"
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    vocab_size = resolve_vocab_size(data_dir, args.vocab_size)

    # Size configurations
    tier_configs = {
        "nano": {"vocab_size": vocab_size, "d_model": 256, "n_layers": 6, "n_heads": 4, "d_ff": 1024, "max_seq_len": args.seq_len},
        "tiny": {"vocab_size": vocab_size, "d_model": 512, "n_layers": 8, "n_heads": 8, "d_ff": 1536, "max_seq_len": args.seq_len},
        "mini": {"vocab_size": vocab_size, "d_model": 768, "n_layers": 12, "n_heads": 12, "d_ff": 2048, "max_seq_len": args.seq_len},
    }
    cfg = tier_configs[args.size]
    model = AdhanModel(**cfg).to(device)
    logger.info(f"Model: {args.size.upper()} initialized with {model.approx_params()/1e6:.2f}M parameters.")

    # Overfit-a-batch sanity test mode
    if args.overfit_batch:
        logger.info("🧪 Running Overfit-a-Batch Sanity Gate (One batch repeated 200 steps)...")
        dummy_x = torch.randint(5, args.vocab_size, (args.batch_size, args.seq_len), device=device)
        dummy_y = torch.randint(5, args.vocab_size, (args.batch_size, args.seq_len), device=device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        
        initial_loss = None
        for step in range(201):
            optimizer.zero_grad()
            logits = model(dummy_x)
            loss = F.cross_entropy(logits.reshape(-1, args.vocab_size), dummy_y.reshape(-1))
            loss.backward()
            optimizer.step()
            if step == 0:
                initial_loss = loss.item()
            if step % 50 == 0:
                logger.info(f"Step {step:3d} | Loss: {loss.item():.4f}")
        
        final_loss = loss.item()
        passed = final_loss < (initial_loss * 0.4)
        status = "PASSED ✅" if passed else "FAILED ❌"
        logger.info(f"Overfit Gate {status} (Loss: {initial_loss:.4f} -> {final_loss:.4f})")
        return

    # Real Pretraining Loop
    train_ds = MemmapShardDataset(train_bin, seq_len=args.seq_len)
    val_ds = MemmapShardDataset(val_bin, seq_len=args.seq_len) if val_bin.exists() else None
    logger.info(f"Loaded {len(train_ds):,} sequences from {train_bin}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95)
    )
    
    # Cosine LR schedule with warmup
    warmup_steps = int(args.max_steps * 0.05)
    def lr_lambda(current_step: int):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, args.max_steps - warmup_steps))
        return max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    autocast_dtype = torch.float16 if compute_dtype == "float16" else torch.bfloat16

    logger.info(f"🚀 Pretraining Started | Micro-batch: {args.batch_size} | Accumulation: {args.grad_accum} (Effective: {args.batch_size * args.grad_accum})")
    
    model.train()
    best_val_loss = float("inf")
    start_time = time.perf_counter()
    tokens_processed = 0

    for step in range(1, args.max_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0

        for _ in range(args.grad_accum):
            x, y = train_ds.get_batch(args.batch_size, device)
            with torch.autocast(device_type=device.type, dtype=autocast_dtype):
                logits = model(x)
                loss = F.cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1))
                loss = loss / args.grad_accum
            
            loss.backward()
            accum_loss += loss.item()
            tokens_processed += args.batch_size * args.seq_len

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        # Logging
        if step % args.log_every == 0 or step == args.max_steps:
            elapsed = time.perf_counter() - start_time
            tok_per_sec = tokens_processed / max(elapsed, 1e-6)
            ppl = math.exp(min(accum_loss, 20.0))
            current_lr = scheduler.get_last_lr()[0]
            logger.info(
                f"Step {step:6d}/{args.max_steps} | Loss: {accum_loss:.4f} | PPL: {ppl:6.2f} | LR: {current_lr:.2e} | Speed: {tok_per_sec:,.0f} tok/s"
            )
            tokens_processed = 0
            start_time = time.perf_counter()

        # Validation & Checkpointing
        if val_ds is not None and (step % args.eval_every == 0 or step == args.max_steps):
            model.eval()
            val_loss = 0.0
            eval_steps = 20
            with torch.no_grad():
                for _ in range(eval_steps):
                    vx, vy = val_ds.get_batch(args.batch_size, device)
                    with torch.autocast(device_type=device.type, dtype=autocast_dtype):
                        v_logits = model(vx)
                        val_loss += F.cross_entropy(v_logits.reshape(-1, vocab_size), vy.reshape(-1)).item()
            
            val_loss /= eval_steps
            val_ppl = math.exp(min(val_loss, 20.0))
            logger.info(f"   📊 [Validation] Loss: {val_loss:.4f} | PPL: {val_ppl:.2f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), ckpt_dir / "adhan_nano_best.pt")
                logger.info(f"   💾 Saved new best model checkpoint -> {ckpt_dir / 'adhan_nano_best.pt'}")
            
            model.train()

    logger.info("🎉 Pretraining Run Complete!")


# --------------------------------------------------------------------------- #
# 5. Direct ONNX Exporter                                                     #
# --------------------------------------------------------------------------- #

def export_onnx(args):
    """Export trained PyTorch checkpoint to ONNX format for Unity / Mobile."""
    ckpt_path = Path(args.checkpoint)
    out_onnx = Path(args.output_onnx or "../yazh-unity/Assets/StreamingAssets/MLModels/yazh-30k.onnx")
    out_onnx.parent.mkdir(parents=True, exist_ok=True)
    
    tier_configs = {
        "nano": {"vocab_size": args.vocab_size, "d_model": 256, "n_layers": 6, "n_heads": 4, "d_ff": 1024, "max_seq_len": args.seq_len},
    }
    model = AdhanModel(**tier_configs[args.size])
    if ckpt_path.exists():
        model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
        logger.info(f"Loaded weights from {ckpt_path}")
    else:
        logger.warning(f"Checkpoint {ckpt_path} not found; exporting initialized model structure.")
    
    model.eval()
    dummy_input = torch.randint(0, args.vocab_size, (1, 64), dtype=torch.int64)
    
    torch.onnx.export(
        model,
        dummy_input,
        str(out_onnx),
        input_names=["input_ids"],
        output_names=["logits"],
        dynamic_axes={"input_ids": {1: "sequence_length"}, "logits": {1: "sequence_length"}},
        opset_version=17
    )
    logger.info(f"✅ Successfully exported ONNX model to {out_onnx}")


def main():
    parser = argparse.ArgumentParser(description="Adhan SLM Efficient Pretraining Engine")
    parser.add_argument("--data-dir", type=str, default="data/final/tamil_slm", help="Directory with train.bin/val.bin")
    parser.add_argument("--size", choices=["nano", "tiny", "mini"], default="nano")
    parser.add_argument("--vocab-size", type=int, default=12000)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8, help="Micro-batch size per forward pass")
    parser.add_argument("--grad-accum", type=int, default=16, help="Gradient accumulation steps")
    parser.add_argument("--max-steps", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=1.5e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/adhan-nano-torch")
    parser.add_argument("--overfit-batch", action="store_true", help="Run overfit sanity gate")
    parser.add_argument("--export-onnx", action="store_true", help="Export checkpoint to ONNX")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/adhan-nano-torch/adhan_nano_best.pt")
    parser.add_argument("--output-onnx", type=str, default=None)
    args = parser.parse_args()

    if args.export_onnx:
        export_onnx(args)
    else:
        train(args)


if __name__ == "__main__":
    main()

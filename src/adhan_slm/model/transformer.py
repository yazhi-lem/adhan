"""Adhan SLM — decoder-only transformer in Flax (agglutination-aware Tamil model).

Architecture (see docs/ARCHITECTURE_SWARAM_SLM.md §3):
  - swaram-vocab token embedding, weight-tied to the output head
  - Rotary position embeddings (RoPE)
  - pre-norm blocks with RMSNorm + SwiGLU MLP
  - optional morpheme-boundary embedding (agglutination signal from tokenizer Layer C)

Three sizes are provided as classmethods: nano (~15M), tiny (~40M), mini (~110M).
Ship nano first (see roadmap Phase 6).

This module imports jax/flax lazily so the rest of the package (tokenizer) works
without the JAX stack installed. Install it with `pip install -r requirements-jax.txt`.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AdhanConfig:
    vocab_size: int = 8000
    d_model: int = 256
    n_layers: int = 6
    n_heads: int = 4
    d_ff: int = 1024          # SwiGLU hidden (per-gate); ~ 8/3 * d_model is typical
    max_seq_len: int = 512
    rope_theta: float = 10000.0
    dropout: float = 0.0
    use_boundary_emb: bool = False   # add morpheme-boundary signal (tokenizer Layer C)
    dtype: str = "bfloat16"          # compute dtype; master weights stay fp32

    @classmethod
    def nano(cls, vocab_size: int = 8000) -> "AdhanConfig":
        return cls(vocab_size, d_model=256, n_layers=6, n_heads=4,
                   d_ff=1024, max_seq_len=512)

    @classmethod
    def tiny(cls, vocab_size: int = 12000) -> "AdhanConfig":
        return cls(vocab_size, d_model=512, n_layers=8, n_heads=8,
                   d_ff=1536, max_seq_len=1024)

    @classmethod
    def mini(cls, vocab_size: int = 16000) -> "AdhanConfig":
        return cls(vocab_size, d_model=768, n_layers=12, n_heads=12,
                   d_ff=2048, max_seq_len=2048)

    def approx_params(self) -> int:
        """Rough parameter count (tied embeddings)."""
        emb = self.vocab_size * self.d_model
        per_layer = (4 * self.d_model * self.d_model          # attn qkvo
                     + 3 * self.d_model * self.d_ff)          # SwiGLU (gate,up,down)
        return emb + self.n_layers * per_layer


def build_model(config: AdhanConfig):
    """Construct the Flax module. Imported lazily to avoid a hard JAX dependency."""
    return AdhanSLM(config)


# --------------------------------------------------------------------------- #
# Flax module. Kept in a factory-style import guard so `from adhan_slm.model    #
# import AdhanConfig` works even when flax is not installed.                    #
# --------------------------------------------------------------------------- #
try:
    import flax.linen as nn
    import jax
    import jax.numpy as jnp

    def _rms_norm(x, weight, eps=1e-6):
        var = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
        return x * jax.lax.rsqrt(var + eps) * weight

    def _rope(x, theta):
        # x: [B, H, T, Dh]
        b, h, t, dh = x.shape
        half = dh // 2
        freqs = 1.0 / (theta ** (jnp.arange(0, half) / half))
        ang = jnp.arange(t)[:, None] * freqs[None, :]          # [T, half]
        cos, sin = jnp.cos(ang), jnp.sin(ang)
        x1, x2 = x[..., :half], x[..., half:]
        rot = jnp.concatenate([x1 * cos - x2 * sin, x1 * sin + x2 * cos], axis=-1)
        return rot

    class Block(nn.Module):
        cfg: AdhanConfig

        @nn.compact
        def __call__(self, x, mask):
            c = self.cfg
            dt = jnp.dtype(c.dtype)
            # --- attention ---
            g1 = self.param("norm1", nn.initializers.ones, (c.d_model,))
            h = _rms_norm(x, g1)
            qkv = nn.Dense(3 * c.d_model, use_bias=False, dtype=dt, name="qkv")(h)
            q, k, v = jnp.split(qkv, 3, axis=-1)
            dh = c.d_model // c.n_heads
            shp = lambda t: t.reshape(t.shape[0], t.shape[1], c.n_heads, dh).transpose(0, 2, 1, 3)
            q, k, v = _rope(shp(q), c.rope_theta), _rope(shp(k), c.rope_theta), shp(v)
            att = (q @ k.transpose(0, 1, 3, 2)) / jnp.sqrt(dh)
            att = jnp.where(mask, att, -1e9)
            att = jax.nn.softmax(att, axis=-1)
            o = (att @ v).transpose(0, 2, 1, 3).reshape(x.shape)
            x = x + nn.Dense(c.d_model, use_bias=False, dtype=dt, name="proj")(o)
            # --- SwiGLU MLP ---
            g2 = self.param("norm2", nn.initializers.ones, (c.d_model,))
            h = _rms_norm(x, g2)
            gate = nn.Dense(c.d_ff, use_bias=False, dtype=dt, name="gate")(h)
            up = nn.Dense(c.d_ff, use_bias=False, dtype=dt, name="up")(h)
            h = nn.Dense(c.d_model, use_bias=False, dtype=dt, name="down")(jax.nn.silu(gate) * up)
            return x + h

    class AdhanSLM(nn.Module):
        cfg: AdhanConfig

        @nn.compact
        def __call__(self, tokens, boundaries=None):
            c = self.cfg
            dt = jnp.dtype(c.dtype)
            emb = self.param("tok_emb", nn.initializers.normal(0.02),
                             (c.vocab_size, c.d_model))
            x = emb[tokens]
            if c.use_boundary_emb and boundaries is not None:
                bmark = self.param("boundary_emb", nn.initializers.normal(0.02),
                                   (2, c.d_model))
                x = x + bmark[boundaries]
            x = x.astype(dt)
            t = tokens.shape[1]
            mask = jnp.tril(jnp.ones((t, t), dtype=bool))[None, None]
            for i in range(c.n_layers):
                x = Block(c, name=f"block_{i}")(x, mask)
            gf = self.param("norm_f", nn.initializers.ones, (c.d_model,))
            x = _rms_norm(x, gf)
            # weight-tied head
            return x.astype(jnp.float32) @ emb.T

except ImportError:  # JAX/Flax not installed — tokenizer-only usage still works.
    class AdhanSLM:  # type: ignore
        def __init__(self, *_a, **_k):
            raise ImportError(
                "AdhanSLM requires jax + flax. Install: pip install -r requirements-jax.txt")


if __name__ == "__main__":
    for name in ("nano", "tiny", "mini"):
        cfg = getattr(AdhanConfig, name)()
        print(f"{name:5s}  d_model={cfg.d_model:4d}  layers={cfg.n_layers:2d}  "
              f"heads={cfg.n_heads:2d}  ctx={cfg.max_seq_len:4d}  "
              f"~{cfg.approx_params()/1e6:.1f}M params")

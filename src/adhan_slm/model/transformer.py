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

    # jax.nn.dot_product_attention (added ~0.4.28) dispatches to a fused cuDNN
    # flash-attention kernel on supported GPUs (implementation=None auto-selects
    # it, falling back to XLA elsewhere) — materializes no O(T^2) score/mask
    # array and is the single biggest single-GPU training throughput lever for
    # a transformer this shape. Guarded so older jax installs still work.
    _HAS_FUSED_ATTN = hasattr(jax.nn, "dot_product_attention")

    def _rms_norm(x, weight, eps=1e-6):
        var = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
        return x * jax.lax.rsqrt(var + eps) * weight

    def _rope(x, theta):
        # x: [B, T, H, Dh] — angle broadcasts over the T axis (axis=1)
        dh = x.shape[-1]
        half = dh // 2
        freqs = 1.0 / (theta ** (jnp.arange(0, half) / half))
        t = x.shape[1]
        ang = jnp.arange(t)[:, None] * freqs[None, :]           # [T, half], fp32
        # cos/sin computed in fp32 for precision, then cast to x's compute dtype
        # (bf16) so q/k/v stay uniform — dot_product_attention requires matching
        # dtypes across q/k/v, and mixed fp32/bf16 here would force an upcast.
        cos = jnp.cos(ang)[None, :, None, :].astype(x.dtype)
        sin = jnp.sin(ang)[None, :, None, :].astype(x.dtype)
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
            b, t = x.shape[0], x.shape[1]
            # (B, T, H, Dh) directly — no transpose needed, matches the layout
            # jax.nn.dot_product_attention expects (and RoPE is axis-agnostic).
            shp = lambda a: a.reshape(b, t, c.n_heads, dh)
            q, k, v = _rope(shp(q), c.rope_theta), _rope(shp(k), c.rope_theta), shp(v)
            if _HAS_FUSED_ATTN:
                o = jax.nn.dot_product_attention(q, k, v, is_causal=True, implementation=None)
            else:
                qT, kT, vT = (a.transpose(0, 2, 1, 3) for a in (q, k, v))
                att = (qT @ kT.transpose(0, 1, 3, 2)) / jnp.sqrt(dh)
                att = jnp.where(mask, att, -1e9)
                att = jax.nn.softmax(att, axis=-1)
                o = (att @ vT).transpose(0, 2, 1, 3)
            o = o.reshape(x.shape)
            x = x + nn.Dense(c.d_model, use_bias=False, dtype=dt, name="proj")(o)
            # --- SwiGLU MLP ---
            g2 = self.param("norm2", nn.initializers.ones, (c.d_model,))
            h = _rms_norm(x, g2)
            gate = nn.Dense(c.d_ff, use_bias=False, dtype=dt, name="gate")(h)
            up = nn.Dense(c.d_ff, use_bias=False, dtype=dt, name="up")(h)
            h = nn.Dense(c.d_model, use_bias=False, dtype=dt, name="down")(jax.nn.silu(gate) * up)
            return x + h

    def _apply_repetition_penalty(logits, prev_tokens, penalty):
        # CTRL-style penalty: divide positive logits / multiply negative ones for
        # already-emitted tokens, discouraging loops (common in small LMs).
        if penalty == 1.0 or prev_tokens.size == 0:
            return logits
        uniq = jnp.unique(prev_tokens, size=prev_tokens.size, fill_value=-1)
        mask = jnp.zeros_like(logits).at[uniq.clip(0)].set(jnp.where(uniq >= 0, 1.0, 0.0))
        pos = logits > 0
        return jnp.where(mask.astype(bool),
                         jnp.where(pos, logits / penalty, logits * penalty), logits)

    def _sample_next(logits, key, temperature, top_k, top_p):
        """Sample one token id from a [vocab] logit vector (greedy if temp==0)."""
        if temperature <= 0.0:
            return jnp.argmax(logits, axis=-1)
        logits = logits / temperature
        vocab = logits.shape[-1]
        if top_k and 0 < top_k < vocab:
            kth = jax.lax.top_k(logits, top_k)[0][..., -1]
            logits = jnp.where(logits < kth, -jnp.inf, logits)
        if top_p and 0.0 < top_p < 1.0:
            order = jnp.argsort(logits)[::-1]
            sorted_logits = logits[order]
            probs = jax.nn.softmax(sorted_logits)
            cum = jnp.cumsum(probs)
            # keep tokens up to and including the one that crosses top_p
            keep = cum - probs < top_p
            sorted_logits = jnp.where(keep, sorted_logits, -jnp.inf)
            logits = logits.at[order].set(sorted_logits)
        return jax.random.categorical(key, logits, axis=-1)

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
            # Fused attention takes is_causal=True and never materializes a mask;
            # only build the O(T^2) boolean array for the XLA fallback path.
            mask = None if _HAS_FUSED_ATTN else jnp.tril(jnp.ones((t, t), dtype=bool))[None, None]
            for i in range(c.n_layers):
                x = Block(c, name=f"block_{i}")(x, mask)
            gf = self.param("norm_f", nn.initializers.ones, (c.d_model,))
            x = _rms_norm(x, gf)
            # weight-tied head
            return x.astype(jnp.float32) @ emb.T

    def generate(model, params, prompt_ids, max_new_tokens=64, temperature=0.8,
                 top_k=40, top_p=0.95, repetition_penalty=1.1, eos_id=None, seed=0):
        """Autoregressive sampling from a trained AdhanSLM.

        `prompt_ids` is a 1-D python list / array of token ids. Returns the full id
        list (prompt + generated). Greedy when ``temperature == 0``. Recomputes the
        forward over the growing prefix each step (no KV-cache yet — fine for a small
        model + short generations; caching is a later throughput optimization).

        This is what Phase 4/5 need: run the kid-level prompt set, feed generations to
        the sandhi probe, and demo the model — none of which was possible before.
        """
        ids = list(int(t) for t in prompt_ids)
        key = jax.random.PRNGKey(seed)
        ctx = model.cfg.max_seq_len
        for _ in range(max_new_tokens):
            window = ids[-ctx:]
            logits = model.apply({"params": params}, jnp.asarray([window], dtype=jnp.int32))
            next_logits = logits[0, -1]
            next_logits = _apply_repetition_penalty(
                next_logits, jnp.asarray(window, dtype=jnp.int32), repetition_penalty)
            key, sub = jax.random.split(key)
            nxt = int(_sample_next(next_logits, sub, temperature, top_k, top_p))
            ids.append(nxt)
            if eos_id is not None and nxt == eos_id:
                break
        return ids

except ImportError:  # JAX/Flax not installed — tokenizer-only usage still works.
    class AdhanSLM:  # type: ignore
        def __init__(self, *_a, **_k):
            raise ImportError(
                "AdhanSLM requires jax + flax. Install: pip install -r requirements-jax.txt")

    def generate(*_a, **_k):  # type: ignore
        raise ImportError(
            "generate() requires jax + flax. Install: pip install -r requirements-jax.txt")


if __name__ == "__main__":
    for name in ("nano", "tiny", "mini"):
        cfg = getattr(AdhanConfig, name)()
        print(f"{name:5s}  d_model={cfg.d_model:4d}  layers={cfg.n_layers:2d}  "
              f"heads={cfg.n_heads:2d}  ctx={cfg.max_seq_len:4d}  "
              f"~{cfg.approx_params()/1e6:.1f}M params")

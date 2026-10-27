"""Adhan SLM — decoder-only transformer in Flax (agglutination-aware Tamil model).

Architecture (see docs/architecture/ARCHITECTURE_SWARAM_SLM.md §3):
  - swaram-vocab token embedding, weight-tied to the output head
  - Rotary position embeddings (RoPE)
  - pre-norm blocks with RMSNorm + SwiGLU MLP
  - optional morpheme-boundary embedding (agglutination signal from tokenizer Layer C)
  - optional Tiṇai conditioning (see docs/architecture/TINAI_CONDITIONING.md and model/tinai.py): a
    causal router over the five Sangam landscapes whose embedding is added to the input
    and modulates every block (FiLM). Off by default; zero-initialised so enabling it
    starts from exactly the unconditioned model.

Three sizes are provided as classmethods: nano (~15M), tiny (~40M), mini (~110M).
Ship nano first (see roadmap Phase 6).

This module imports jax/flax lazily so the rest of the package (tokenizer) works
without the JAX stack installed. Install it with `pip install -r requirements-jax.txt`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .tinai import N_TINAI, tinai_index


@dataclass
class AdhanConfig:
    vocab_size: int = 8000
    d_model: int = 256
    n_layers: int = 6
    n_heads: int = 4
    # Grouped-query attention: number of K/V heads. 0 (the default) means
    # n_kv_heads == n_heads, i.e. plain multi-head attention, so existing
    # checkpoints keep loading with the same 'qkv' kernel shape. n_kv_heads <
    # n_heads shares each K/V head across n_heads // n_kv_heads query heads,
    # which cuts the K/V projections and the KV memory of the attention step —
    # the part that actually hurts at long context on a 16 GB T4.
    n_kv_heads: int = 0
    d_ff: int = 1024  # SwiGLU hidden (per-gate); ~ 8/3 * d_model is typical
    max_seq_len: int = 512
    rope_theta: float = 10000.0
    # Applied to the embedding, the attention output and the MLP output. 0.1 is
    # the usual pretraining default; only active when the caller passes
    # deterministic=False (training), never during eval/generation.
    dropout: float = 0.0
    use_boundary_emb: bool = False  # add morpheme-boundary signal (tokenizer Layer C)
    use_tinai: bool = False  # Tiṇai (five-landscape) conditioning layer; off = old checkpoints load
    dtype: str = "bfloat16"  # compute dtype; master weights stay fp32
    # Gradient checkpointing (jax.remat): trades compute for memory by not
    # storing block activations during the forward pass. Essential for training
    # mini (12-layer, 2048 ctx) on a single TPU/GPU. No effect at inference.
    remat: bool = False

    @property
    def kv_heads(self) -> int:
        """Effective K/V head count (0 in the config means classic MHA)."""
        return self.n_kv_heads or self.n_heads

    def __post_init__(self) -> None:
        if not isinstance(self.vocab_size, int) or self.vocab_size <= 0:
            raise ValueError(f"vocab_size must be a positive int, got {self.vocab_size}")
        if not isinstance(self.d_model, int) or self.d_model <= 0:
            raise ValueError(f"d_model must be a positive int, got {self.d_model}")
        if not isinstance(self.n_layers, int) or self.n_layers <= 0:
            raise ValueError(f"n_layers must be a positive int, got {self.n_layers}")
        if not isinstance(self.n_heads, int) or self.n_heads <= 0:
            raise ValueError(f"n_heads must be a positive int, got {self.n_heads}")
        if self.n_heads % self.kv_heads:
            raise ValueError(
                f"n_heads={self.n_heads} must be divisible by n_kv_heads={self.kv_heads}"
            )
        if self.d_model % self.n_heads:
            raise ValueError(f"d_model={self.d_model} must be divisible by n_heads={self.n_heads}")
        if self.d_ff <= 0:
            raise ValueError(f"d_ff must be positive, got {self.d_ff}")
        if self.max_seq_len <= 0:
            raise ValueError(f"max_seq_len must be positive, got {self.max_seq_len}")
        if self.dropout < 0.0 or self.dropout > 1.0:
            raise ValueError(f"dropout must be in [0, 1], got {self.dropout}")
        valid_dtypes = ("float32", "float64", "bfloat16", "float16")
        if self.dtype not in valid_dtypes:
            raise ValueError(f"dtype must be one of {valid_dtypes}, got {self.dtype!r}")

    @classmethod
    def nano(cls, vocab_size: int = 8000) -> "AdhanConfig":
        return cls(vocab_size, d_model=256, n_layers=6, n_heads=4, d_ff=1024, max_seq_len=512)

    @classmethod
    def tiny(cls, vocab_size: int = 12000) -> "AdhanConfig":
        return cls(vocab_size, d_model=512, n_layers=8, n_heads=8, d_ff=1536, max_seq_len=1024)

    @classmethod
    def mini(cls, vocab_size: int = 16000) -> "AdhanConfig":
        return cls(vocab_size, d_model=768, n_layers=12, n_heads=12, d_ff=2048, max_seq_len=2048)

    def approx_params(self) -> int:
        """Rough parameter count (tied embeddings), GQA-aware."""
        emb = self.vocab_size * self.d_model
        dh = self.d_model // self.n_heads
        # q projects to d_model; k and v each to kv_heads * head_dim; o back to d_model
        attn = 2 * self.d_model * self.d_model + 2 * self.d_model * self.kv_heads * dh
        mlp = 3 * self.d_model * self.d_ff  # SwiGLU gate, up, down
        norms = 2 * self.d_model  # norm1 + norm2 per block
        return emb + self.n_layers * (attn + mlp + norms) + self.d_model  # + norm_f


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

    def _rope(x, theta, offset=0):
        # x: [B, T, H, Dh] — angle broadcasts over the T axis (axis=1).
        # ``offset`` is the absolute position of the first token in ``x``; it is 0
        # for a full-sequence forward and ``past_len`` during cached decoding.
        dh = x.shape[-1]
        half = dh // 2
        freqs = 1.0 / (theta ** (jnp.arange(0, half) / half))
        t = x.shape[1]
        # ``offset + arange`` keeps this traceable when offset is a jit argument.
        pos = offset + jnp.arange(t)
        ang = pos[:, None] * freqs[None, :]  # [T, half], fp32
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
        def __call__(self, x, mask, cond=None, deterministic=True, cache=None, offset=0):
            """One transformer block.

            ``cache`` is ``None`` for the training/full-sequence path, or a
            ``(k_past, v_past)`` pair during cached decoding. When a cache is
            given the block returns ``(x, (k_new, v_new))`` so the caller can
            carry it to the next step.
            """
            c = self.cfg
            dt = jnp.dtype(c.dtype)
            # Tiṇai FiLM: per-position scale/shift for both pre-norms, zero-initialised so
            # the block is the identity modulation until training moves it.
            s1 = t1 = s2 = t2 = 0.0
            if c.use_tinai and cond is not None:
                film = nn.Dense(
                    4 * c.d_model,
                    use_bias=False,
                    kernel_init=nn.initializers.zeros,
                    dtype=dt,
                    name="film",
                )(cond.astype(dt))
                s1, t1, s2, t2 = jnp.split(film, 4, axis=-1)
            # --- attention ---
            g1 = self.param("norm1", nn.initializers.ones, (c.d_model,))
            h = _rms_norm(x, g1) * (1.0 + s1) + t1

            # One fused projection for q and the (smaller) k/v stack. With
            # n_kv_heads == n_heads this is exactly the old 3*d_model "qkv" kernel,
            # so multi-head checkpoints load unchanged; with GQA it is
            # (n_heads + 2*kv_heads) * head_dim wide and the k/v part is smaller.
            dh = c.d_model // c.n_heads
            kvh = c.kv_heads
            qkv = nn.Dense((c.n_heads + 2 * kvh) * dh, use_bias=False, dtype=dt, name="qkv")(h)
            q, k, v = jnp.split(qkv, [c.n_heads * dh, (c.n_heads + kvh) * dh], axis=-1)
            b, t = x.shape[0], x.shape[1]

            # (B, T, H, Dh) directly — no transpose needed, matches the layout
            # jax.nn.dot_product_attention expects (and RoPE is axis-agnostic).
            # RoPE is applied before the K/V heads are broadcast, so the rotation
            # costs nothing extra under GQA.
            q = _rope(q.reshape(b, t, c.n_heads, dh), c.rope_theta, offset=offset)
            k = _rope(k.reshape(b, t, kvh, dh), c.rope_theta, offset=offset)
            v = v.reshape(b, t, kvh, dh)

            # --- KV cache (fixed-size buffers) ---
            # The cache holds K/V in [B, S_max, kvh, dh] buffers that never change
            # shape, so a compiled decode step stays valid for every token. New
            # K/V is written at ``offset`` with a dynamic update; padding beyond
            # the valid length is masked out in the attention step.
            new_cache = None
            attn_mask = None
            if cache is not None:
                k_buf, v_buf = cache
                start = (0, offset, 0, 0)
                k_buf = jax.lax.dynamic_update_slice(k_buf, k, start)
                v_buf = jax.lax.dynamic_update_slice(v_buf, v, start)
                new_cache = (k_buf, v_buf)
                k_full, v_full = k_buf, v_buf
                s_max = k_buf.shape[1]
                qpos = offset + jnp.arange(t)
                kpos = jnp.arange(s_max)
                # [1, 1, t, S_max]: each query sees keys up to its own position.
                attn_mask = (kpos[None, :] <= qpos[:, None])[None, None, :, :]
            else:
                k_full, v_full = k, v
            t_full = k_full.shape[1]

            if kvh != c.n_heads:
                # --- P1: Memory-efficient GQA head broadcast ---
                # Instead of materializing a repeated K/V array (jnp.repeat),
                # reshape to add a unit axis and let XLA broadcast. This keeps
                # the K/V activation memory at kv_heads instead of n_heads.
                groups = c.n_heads // kvh
                k_full = jnp.broadcast_to(
                    k_full[:, :, :, None, :], (b, t_full, kvh, groups, dh)
                ).reshape(b, t_full, c.n_heads, dh)
                v_full = jnp.broadcast_to(
                    v_full[:, :, :, None, :], (b, t_full, kvh, groups, dh)
                ).reshape(b, t_full, c.n_heads, dh)

            if cache is not None:
                # Cached path always uses an explicit mask (causal-combined).
                if _HAS_FUSED_ATTN:
                    o = jax.nn.dot_product_attention(
                        q, k_full, v_full, mask=attn_mask, is_causal=False, implementation=None
                    )
                else:
                    qT, kT, vT = (a.transpose(0, 2, 1, 3) for a in (q, k_full, v_full))
                    att = (qT @ kT.transpose(0, 1, 3, 2)) / jnp.sqrt(dh)
                    att = jnp.where(attn_mask, att, -1e9)
                    att = jax.nn.softmax(att, axis=-1)
                    o = (att @ vT).transpose(0, 2, 1, 3)
            elif _HAS_FUSED_ATTN:
                o = jax.nn.dot_product_attention(
                    q, k_full, v_full, is_causal=True, implementation=None
                )
            else:
                qT, kT, vT = (a.transpose(0, 2, 1, 3) for a in (q, k_full, v_full))
                att = (qT @ kT.transpose(0, 1, 3, 2)) / jnp.sqrt(dh)
                att = jnp.where(mask, att, -1e9)
                att = jax.nn.softmax(att, axis=-1)
                o = (att @ vT).transpose(0, 2, 1, 3)
            o = o.reshape(x.shape)
            # Dropout on the attention output: cheap (one tensor per block) and it
            # regularises the residual stream both axes actually write to.
            o = nn.Dropout(rate=c.dropout, deterministic=deterministic)(o)
            x = x + nn.Dense(c.d_model, use_bias=False, dtype=dt, name="proj")(o)
            # --- SwiGLU MLP ---
            g2 = self.param("norm2", nn.initializers.ones, (c.d_model,))
            h = _rms_norm(x, g2) * (1.0 + s2) + t2
            gate = nn.Dense(c.d_ff, use_bias=False, dtype=dt, name="gate")(h)
            up = nn.Dense(c.d_ff, use_bias=False, dtype=dt, name="up")(h)
            h = nn.Dense(c.d_model, use_bias=False, dtype=dt, name="down")(jax.nn.silu(gate) * up)
            h = nn.Dropout(rate=c.dropout, deterministic=deterministic)(h)
            out = x + h
            if cache is not None:
                return out, new_cache
            return out

    def _apply_repetition_penalty(logits, prev_tokens, penalty):
        # CTRL-style penalty: divide positive logits / multiply negative ones for
        # already-emitted tokens, discouraging loops (common in small LMs).
        if penalty == 1.0 or prev_tokens.size == 0:
            return logits
        uniq = jnp.unique(prev_tokens, size=prev_tokens.size, fill_value=-1)
        mask = jnp.zeros_like(logits).at[uniq.clip(0)].set(jnp.where(uniq >= 0, 1.0, 0.0))
        pos = logits > 0
        return jnp.where(
            mask.astype(bool), jnp.where(pos, logits / penalty, logits * penalty), logits
        )

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
        def __call__(
            self,
            tokens: jax.Array,
            boundaries: jax.Array | None = None,
            tinai: jax.Array | None = None,
            return_router: bool = False,
            deterministic: bool = True,
            cache: list | None = None,
            offset: int = 0,
            return_cache: bool = False,
        ):
            """Logits ``[B, T, vocab]``.

            ``tinai`` (only when ``cfg.use_tinai``): per-example landscape class ``[B]``
            (or a single int), ``-1`` = let the router decide. ``return_router=True``
            returns ``(logits, router_logits[B, T, 5])`` for the auxiliary loss.

            ``cache`` / ``offset`` / ``return_cache`` enable KV-cached decoding:
            ``cache`` is a list of per-layer ``(k, v)`` arrays (or ``None`` entries),
            ``offset`` is the absolute position of the first input token, and
            ``return_cache=True`` makes the call return ``(logits, new_cache)``.
            Cached decoding is not supported together with ``use_tinai`` (the
            router pools the full prefix, which a single-token step cannot see).
            """
            c = self.cfg
            if cache is not None and c.use_tinai:
                raise ValueError(
                    "KV-cached decoding is not supported with use_tinai=True; "
                    "build the model with use_tinai=False for fast generation."
                )
            dt = jnp.dtype(c.dtype)
            emb = self.param("tok_emb", nn.initializers.normal(0.02), (c.vocab_size, c.d_model))
            x = emb[tokens]
            if c.use_boundary_emb and boundaries is not None:
                bmark = self.param("boundary_emb", nn.initializers.normal(0.02), (2, c.d_model))
                x = x + bmark[boundaries]
            x = x.astype(dt)
            x = nn.Dropout(rate=c.dropout, deterministic=deterministic)(x)
            t = tokens.shape[1]
            cond = router_logits = None
            if c.use_tinai:
                b = tokens.shape[0]
                # Causal router: a running mean of the embeddings, so the prediction at
                # position t sees only tokens <= t (safe for autoregressive generation).
                counts = jnp.arange(1, t + 1, dtype=jnp.float32)[None, :, None]
                pooled = jnp.cumsum(x.astype(jnp.float32), axis=1) / counts
                router_logits = nn.Dense(N_TINAI, dtype=jnp.float32, name="tinai_router")(pooled)
                weights = jax.nn.softmax(router_logits, axis=-1)  # [B, T, 5]
                if tinai is not None:
                    lab = jnp.broadcast_to(jnp.asarray(tinai, dtype=jnp.int32), (b,))
                    hard = jax.nn.one_hot(jnp.clip(lab, 0, N_TINAI - 1), N_TINAI)[:, None, :]
                    weights = jnp.where((lab >= 0)[:, None, None], hard, weights)
                tinai_emb = self.param(
                    "tinai_emb", nn.initializers.normal(0.02), (N_TINAI, c.d_model)
                )
                # Zero-initialised gate: enabling the layer starts as the plain model.
                gate = self.param("tinai_gate", nn.initializers.zeros, ())
                cond = (weights @ tinai_emb) * gate  # [B, T, d_model], fp32
                x = x + cond.astype(dt)
            # Fused attention takes is_causal=True and never materializes a mask;
            # only build the O(T^2) boolean array for the XLA fallback path.
            mask = None if _HAS_FUSED_ATTN else jnp.tril(jnp.ones((t, t), dtype=bool))[None, None]

            # --- P1: Gradient checkpointing (remat) ---
            block_fn = Block(c)
            if c.remat and cache is None:
                block_fn = jax.checkpoint(block_fn, policy=jax.checkpoint_policies.dots_saveable)

            new_cache = [] if return_cache else None
            # On prefill (cache is None but return_cache is set) allocate the
            # fixed-size K/V buffers once. Their shape never changes afterwards,
            # which is what lets a compiled decode step be reused for every token.
            working_cache = cache
            if return_cache and working_cache is None:
                bsz = tokens.shape[0]
                dh_ = c.d_model // c.n_heads
                buf = jnp.zeros((bsz, c.max_seq_len, c.kv_heads, dh_), dtype=dt)
                working_cache = [(buf, buf) for _ in range(c.n_layers)]
            for i in range(c.n_layers):
                if not return_cache:
                    x = block_fn(x, mask, cond, deterministic=deterministic)
                else:
                    x, layer_cache_new = block_fn(
                        x,
                        mask,
                        cond,
                        deterministic=deterministic,
                        cache=working_cache[i],
                        offset=offset,
                    )
                    new_cache.append(layer_cache_new)
            gf = self.param("norm_f", nn.initializers.ones, (c.d_model,))
            x = _rms_norm(x, gf)
            # weight-tied head
            logits = x.astype(jnp.float32) @ emb.T
            if return_cache:
                return logits, new_cache
            return (logits, router_logits) if return_router else logits

    def tinai_aux_loss(router_logits, labels):
        """Cross-entropy of the router against weak tiṇai labels.

        ``router_logits`` is ``[B, T, 5]`` from ``model.apply(..., return_router=True)``;
        the last position has seen the whole sequence, so that is the one supervised.
        ``labels`` is ``[B]`` with ``-1`` (unspecified) ignored. Returns 0 if none labelled.
        """
        labels = jnp.asarray(labels, dtype=jnp.int32)
        logp = jax.nn.log_softmax(router_logits[:, -1, :], axis=-1)
        nll = -jnp.take_along_axis(logp, jnp.clip(labels, 0, N_TINAI - 1)[:, None], axis=-1)[:, 0]
        valid = labels >= 0
        return jnp.where(valid, nll, 0.0).sum() / jnp.maximum(valid.sum(), 1)

    def generate(
        model: AdhanSLM,
        params: Any,
        prompt_ids: list[int],
        max_new_tokens: int = 64,
        temperature: float = 0.8,
        top_k: int = 40,
        top_p: float = 0.95,
        repetition_penalty: float = 1.1,
        eos_id: int | None = None,
        seed: int = 0,
        tinai: int | str | None = None,
    ) -> list[int]:
        """Autoregressive sampling from a trained AdhanSLM.

        `prompt_ids` is a 1-D python list / array of token ids. Returns the full id
        list (prompt + generated). Greedy when ``temperature == 0``. Recomputes the
        forward over the growing prefix each step (no KV-cache yet — fine for a small
        model + short generations; caching is a later throughput optimization).

        ``tinai`` steers generation toward one of the five landscapes (an index 0-4 or a
        key like ``"mullai"``); it needs a model built with ``use_tinai=True``.
        """
        if tinai is not None:
            if not model.cfg.use_tinai:
                raise ValueError("tinai= needs a model built with AdhanConfig(use_tinai=True)")
            if isinstance(tinai, str):
                tinai = tinai_index(tinai)
            if not 0 <= int(tinai) < N_TINAI:
                raise ValueError(f"tinai must be 0..{N_TINAI - 1}, got {tinai}")
        ids = list(int(t) for t in prompt_ids)
        key = jax.random.PRNGKey(seed)
        ctx = model.cfg.max_seq_len
        # KV-cache is used whenever the model does not need Tiṇai conditioning
        # (the router pools the whole prefix and cannot run one token at a time).
        use_cache = not model.cfg.use_tinai
        cache = None
        offset = 0
        pparams = {"params": params}

        # Two compiled programs only: prefill (prompt-length) and decode (one
        # token). Fixed-size cache buffers keep the decode shape stable, so the
        # second and later tokens reuse the same executable.
        if use_cache:

            @jax.jit
            def _prefill(p, toks):
                return model.apply(p, toks, cache=None, offset=jnp.asarray(0), return_cache=True)

            @jax.jit
            def _decode(p, tok, cache_in, off):
                return model.apply(p, tok, cache=cache_in, offset=off, return_cache=True)

        for _ in range(max_new_tokens):
            if use_cache:
                if cache is None:
                    feed = ids[-ctx:]
                    offset = 0
                    logits, cache = _prefill(pparams, jnp.asarray([feed], dtype=jnp.int32))
                    offset += len(feed)
                else:
                    logits, cache = _decode(
                        pparams,
                        jnp.asarray([[ids[-1]]], dtype=jnp.int32),
                        cache,
                        jnp.asarray(offset),
                    )
                    offset += 1
                next_logits = logits[0, -1]
            else:
                window = ids[-ctx:]
                extra = {} if tinai is None else {"tinai": int(tinai)}
                logits = model.apply(
                    {"params": params}, jnp.asarray([window], dtype=jnp.int32), **extra
                )
                next_logits = logits[0, -1]
            next_logits = _apply_repetition_penalty(
                next_logits,
                jnp.asarray(ids[-ctx:], dtype=jnp.int32),
                repetition_penalty,
            )
            key, sub = jax.random.split(key)
            nxt = int(_sample_next(next_logits, sub, temperature, top_k, top_p))
            ids.append(nxt)
            if offset > ctx:
                # Keep the cache from growing past the model's context window by
                # resetting it; the next step re-prefills from the last ctx tokens.
                cache = None
                offset = 0
            if eos_id is not None and nxt == eos_id:
                break
        return ids

except ImportError:  # JAX/Flax not installed — tokenizer-only usage still works.

    class AdhanSLM:  # type: ignore
        def __init__(self, *_a, **_k):
            raise ImportError(
                "AdhanSLM requires jax + flax. Install: pip install -r requirements-jax.txt"
            )

    def generate(*_a, **_k):  # type: ignore
        raise ImportError(
            "generate() requires jax + flax. Install: pip install -r requirements-jax.txt"
        )

    def tinai_aux_loss(*_a, **_k):  # type: ignore
        raise ImportError(
            "tinai_aux_loss() requires jax + flax. Install: pip install -r requirements-jax.txt"
        )


if __name__ == "__main__":
    from adhan_slm.core.logging import get_logger

    logger = get_logger(__name__)
    for name in ("nano", "tiny", "mini"):
        cfg = getattr(AdhanConfig, name)()
        logger.info(
            f"{name:5s}  d_model={cfg.d_model:4d}  layers={cfg.n_layers:2d}  "
            f"heads={cfg.n_heads:2d}  ctx={cfg.max_seq_len:4d}  "
            f"~{cfg.approx_params()/1e6:.1f}M params"
        )

"""Load a trained Adhan SLM (tokenizer + Orbax checkpoint) for generation / eval.

Ties together the three artifacts a run produces:
  * ``vocab.json`` + ``merges.txt``  — frozen swaram tokenizer (prepare_slm_corpus)
  * an Orbax checkpoint dir           — trained params (train_jax, Phase 3)
  * the training YAML                 — model size / vocab, to rebuild the architecture

Everything JAX-side is imported lazily so importing this module never forces the JAX
stack; ``load_model`` raises a clear message if it's missing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import yaml

from adhan_slm.model import AdhanConfig
from adhan_slm.tokenizer import SwaramTokenizer
from adhan_slm.tokenizer.aksharam_tokenizer import AksharamTokenizer

_TOKENIZERS = {"swaram": SwaramTokenizer, "aksharam": AksharamTokenizer}


def load_tokenizer(tokenizer_dir: str | Path, kind: str = "swaram"):
    d = Path(tokenizer_dir)
    return _TOKENIZERS[kind].from_files(str(d / "vocab.json"), str(d / "merges.txt"))


def _config_from_yaml(config_path: str | Path, vocab_size: Optional[int] = None) -> AdhanConfig:
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    m = cfg.get("model", {})
    size = m.get("size")
    vocab = vocab_size or m.get("vocab_size", 8000)
    if size in ("nano", "tiny", "mini"):
        return getattr(AdhanConfig, size)(vocab_size=vocab)
    return AdhanConfig(**{k: v for k, v in m.items() if k != "size"})


def load_checkpoint_metadata(checkpoint_dir, step: Optional[int] = None) -> dict:
    """Read back the JSON ``metadata`` item train_jax saves alongside a checkpoint
    (model_config + tokenizer_dir) — the "metadata transport" that makes a
    checkpoint dir self-describing instead of needing a separately-passed
    --config/--tokenizer-dir. Raises FileNotFoundError if the checkpoint (or its
    metadata item, for checkpoints saved before this existed) isn't present.
    """
    import orbax.checkpoint as ocp

    mgr = ocp.CheckpointManager(str(Path(checkpoint_dir).resolve()))
    step = step if step is not None else mgr.latest_step()
    if step is None:
        raise FileNotFoundError(f"no checkpoint found in {checkpoint_dir}")
    restored = mgr.restore(step, args=ocp.args.Composite(metadata=ocp.args.JsonRestore()))
    metadata = restored.get("metadata")
    if not metadata:
        raise FileNotFoundError(
            f"checkpoint {checkpoint_dir} (step {step}) has no metadata item — "
            "it predates self-describing checkpoints; pass --config explicitly."
        )
    return metadata


def load_model(
    config_path: Optional[str | Path] = None,
    checkpoint_dir: Optional[str | Path] = None,
    vocab_size: Optional[int] = None,
    step: Optional[int] = None,
):
    """Rebuild the model and restore params from the latest (or given) Orbax step.

    ``config_path=None`` pulls the architecture out of the checkpoint's own
    metadata item instead (see ``load_checkpoint_metadata``) — the normal path
    for checkpoints trained after metadata transport was added; pass an explicit
    YAML ``config_path`` for older checkpoints that don't have one.

    Returns ``(model, params, model_cfg)``.
    """
    if checkpoint_dir is None:
        raise ValueError("load_model requires checkpoint_dir")
    try:
        import jax.numpy as jnp
        import orbax.checkpoint as ocp

        from adhan_slm.model import AdhanSLM
    except ImportError as e:
        raise ImportError(
            f"load_model needs the JAX stack ({e}). pip install -r requirements-jax.txt"
        )

    if config_path is not None:
        model_cfg = _config_from_yaml(config_path, vocab_size=vocab_size)
    else:
        meta = load_checkpoint_metadata(checkpoint_dir, step=step)
        model_cfg = AdhanConfig(**meta["model_config"])
        if vocab_size is not None:
            model_cfg.vocab_size = vocab_size

    model = AdhanSLM(model_cfg)

    # Restore the params-only checkpoint item (train_jax saves it alongside the full
    # state for exactly this reason) — no optimizer pytree to reconstruct.
    import jax

    key = jax.random.PRNGKey(0)
    dummy = jnp.ones((1, min(8, model_cfg.max_seq_len)), dtype=jnp.int32)
    params_template = model.init(key, dummy)["params"]

    mgr = ocp.CheckpointManager(str(Path(checkpoint_dir).resolve()))
    step = step if step is not None else mgr.latest_step()
    if step is None:
        raise FileNotFoundError(f"no checkpoint found in {checkpoint_dir}")
    restored = mgr.restore(
        step, args=ocp.args.Composite(params=ocp.args.StandardRestore(params_template))
    )
    return model, restored["params"], model_cfg


def generate_text(model, params, tokenizer, prompt: str, **gen_kw) -> str:
    """Encode a prompt, sample, and decode back to Tamil text."""
    from adhan_slm.model import generate

    eos = tokenizer.vocab.get("<eos>")
    prompt_ids = tokenizer.encode(prompt, add_special=True)
    # drop a trailing <eos> so generation continues the prompt
    if eos is not None and prompt_ids and prompt_ids[-1] == eos:
        prompt_ids = prompt_ids[:-1]
    out_ids = generate(model, params, prompt_ids, eos_id=eos, **gen_kw)
    return tokenizer.decode(out_ids)

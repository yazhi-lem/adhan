"""Adhan SLM pretraining loop in JAX/Flax with MLflow tracking.

This is the Phase 0 skeleton: a runnable, jit-ed causal-LM training step wired to
MLflow. `--smoke` trains a few steps on synthetic data to prove the loop end-to-end
without a corpus; a real run streams packed swaram-tokenized shards (Phase 2/3).

    # smoke test (needs requirements-jax.txt installed)
    python -m adhan_slm.training.train_jax \
        --config src/adhan_slm/configs/adhan_slm_tiny.yaml --smoke

Full runs plug a Grain/tf.data iterator into `data_iterator()` and enable Orbax
checkpointing (marked TODO below).
"""
from __future__ import annotations

import argparse
import functools
import math
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # src/ on path

from adhan_slm.model import AdhanConfig  # noqa: E402
from adhan_slm.training.mlflow_utils import track_run  # noqa: E402


def load_config(path: str) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def _config_from_dict(d: dict) -> AdhanConfig:
    size = d.get("model", {}).get("size")
    vocab = d.get("model", {}).get("vocab_size", 8000)
    if size in ("nano", "tiny", "mini"):
        return getattr(AdhanConfig, size)(vocab_size=vocab)
    return AdhanConfig(**{k: v for k, v in d.get("model", {}).items() if k != "size"})


def synthetic_batches(vocab_size, batch, seq_len, n, seed=0):
    """Random token batches for smoke-testing the loop (no corpus needed)."""
    import numpy as np
    rng = np.random.default_rng(seed)
    for _ in range(n):
        yield rng.integers(5, vocab_size, size=(batch, seq_len), dtype="int32")


def _resolve(path: str) -> Path:
    """Resolve a config path relative to the repo root if not absolute."""
    p = Path(path)
    return p if p.is_absolute() else Path(__file__).resolve().parents[3] / p


def data_iterator(cfg: dict, batch_size: int, seed: int = 0, infinite: bool = True):
    """Stream packed swaram-tokenized shards for a real run (roadmap Phase 2/3).

    Reads the packed ``train.bin`` shard produced by ``scripts/prepare_slm_corpus.py``
    (path from ``cfg['data']['shards']``) via the pure-python/numpy data loader and
    yields ``(batch, seq_len)`` int32 batches. Replaces the old NotImplementedError
    stub — this is the concrete Grain-free packing loader described in the roadmap.
    """
    from adhan_slm.data import PackedDataset
    from adhan_slm.data.packing import load_manifest

    shards = cfg.get("data", {}).get("shards")
    if not shards:
        raise ValueError("config data.shards is required for a non-smoke run")
    shard_dir = _resolve(shards)
    train_bin = shard_dir / "train.bin" if shard_dir.is_dir() else shard_dir
    if not train_bin.exists():
        raise FileNotFoundError(
            f"packed shard not found: {train_bin}. Build it first:\n"
            f"  python scripts/prepare_slm_corpus.py --corpus <path> --out {shard_dir}")
    manifest = load_manifest(train_bin)
    print(f"[train_jax] shard {train_bin.name}: {manifest.n_sequences:,} seqs "
          f"× {manifest.seq_len} tok  ({manifest.n_tokens:,} tokens, {manifest.dtype})")
    return PackedDataset.from_shard(train_bin, batch_size, manifest=manifest,
                                    seed=seed, infinite=infinite)


def _frozen_vocab_size(cfg: dict):
    """Size of the frozen tokenizer's vocab.json in the shard dir, or None."""
    shards = cfg.get("data", {}).get("shards")
    if not shards:
        return None
    shard_dir = _resolve(shards)
    vocab_json = (shard_dir / "vocab.json") if shard_dir.is_dir() else None
    if vocab_json is None or not vocab_json.exists():
        return None
    import json
    return len(json.loads(vocab_json.read_text(encoding="utf-8")))


def _val_iterator(cfg: dict, batch_size: int):
    """One-pass validation loader over ``val.bin`` if present, else None."""
    from adhan_slm.data import PackedDataset
    from adhan_slm.data.packing import load_manifest

    shards = cfg.get("data", {}).get("shards")
    if not shards:
        return None
    shard_dir = _resolve(shards)
    val_bin = shard_dir / "val.bin" if shard_dir.is_dir() else None
    if val_bin is None or not val_bin.exists():
        return None
    manifest = load_manifest(val_bin)
    return PackedDataset.from_shard(val_bin, batch_size, manifest=manifest,
                                    shuffle=False, infinite=False, drop_last=False)


def train(config_path: str, smoke: bool = False):
    cfg = load_config(config_path)
    model_cfg = _config_from_dict(cfg)
    tcfg = cfg.get("train", {})

    try:
        import jax
        import jax.numpy as jnp
        import optax
        from flax.training import train_state
        from adhan_slm.model import AdhanSLM
    except ImportError as e:
        print(f"[train_jax] JAX stack not available ({e}).")
        print("Install: pip install -r requirements-jax.txt")
        print(f"\nWould train {model_cfg} (~{model_cfg.approx_params()/1e6:.1f}M params).")
        return

    steps = 20 if smoke else int(tcfg.get("max_steps", 100000))
    batch = 4 if smoke else int(tcfg.get("batch_size", 32))
    lr = float(tcfg.get("learning_rate", 3e-4))
    warmup = int(tcfg.get("warmup_steps", 2000))
    seed = int(tcfg.get("seed", 0))

    # For real runs the sequence length is fixed by the packed shard; smoke uses a
    # short synthetic length. Build the data source before init so dummy shapes match.
    if smoke:
        seq_len = 128
        batches = synthetic_batches(model_cfg.vocab_size, batch, seq_len, steps)
        val_ds = None
    else:
        # The model's vocab MUST equal the frozen tokenizer's size, or the embedding
        # rows won't line up with the ids in the shards. The trained merge count rarely
        # hits the configured target exactly, so trust vocab.json over the YAML.
        frozen = _frozen_vocab_size(cfg)
        if frozen is not None and frozen != model_cfg.vocab_size:
            print(f"[train_jax] vocab_size {model_cfg.vocab_size} (config) -> "
                  f"{frozen} (frozen tokenizer) to match shards")
            model_cfg.vocab_size = frozen
        train_ds = data_iterator(cfg, batch, seed=seed, infinite=True)
        seq_len = train_ds.seq_len
        batches = iter(train_ds)
        val_ds = _val_iterator(cfg, batch)

    model = AdhanSLM(model_cfg)
    key = jax.random.PRNGKey(int(tcfg.get("seed", 0)))
    dummy = jnp.ones((batch, seq_len), dtype=jnp.int32)
    params = model.init(key, dummy)["params"]

    # warmup_steps must be strictly less than decay_steps or optax's internal
    # cosine phase length (decay_steps - warmup_steps) hits zero and raises —
    # bites `--smoke` runs where warmup_steps (config default 2000) otherwise
    # clamps equal to the tiny smoke step count.
    warmup_steps = min(warmup, max(1, steps - 1))
    sched = optax.warmup_cosine_decay_schedule(
        0.0, lr, warmup_steps=warmup_steps, decay_steps=steps, end_value=lr * 0.1)
    tx = optax.chain(optax.clip_by_global_norm(1.0),
                     optax.adamw(sched, weight_decay=float(tcfg.get("weight_decay", 0.1))))
    state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

    def loss_fn(params, batch):
        logits = model.apply({"params": params}, batch[:, :-1])
        targets = batch[:, 1:]
        ll = optax.softmax_cross_entropy_with_integer_labels(logits, targets)
        return ll.mean()

    # donate_argnums=(0,): let XLA reuse the input `state`'s device buffers for
    # the output instead of allocating a fresh copy every step. Safe because the
    # loop below always rebinds `state = ...` and never reads the pre-step value
    # again — a standard JAX/Flax training-loop memory/throughput optimization.
    @functools.partial(jax.jit, donate_argnums=(0,))
    def train_step(state, batch):
        loss, grads = jax.value_and_grad(loss_fn)(state.params, batch)
        return state.apply_gradients(grads=grads), loss

    # A batched, jit-ed eval step over the (finite) validation loader. Kept separate
    # from train_step so it never touches optimizer state.
    @jax.jit
    def eval_step(params, batch):
        return loss_fn(params, batch)

    def validate(params, max_batches: int = 200):
        """Mean val loss / per-token perplexity over up to `max_batches` val batches."""
        if val_ds is None:
            return None, None
        total, n = 0.0, 0
        for i, vb in enumerate(val_ds):
            if i >= max_batches:
                break
            total += float(eval_step(params, jnp.asarray(vb)))
            n += 1
        if n == 0:
            return None, None
        mean = total / n
        return mean, math.exp(min(mean, 20.0))

    # Orbax checkpointing (roadmap Phase 3): async, resumable, best-by-val. Degrades
    # to a no-op with a warning if orbax isn't installed, so training still runs.
    ckpt_dir = cfg.get("checkpoint_dir")
    ckptr = None
    if ckpt_dir and not smoke:
        try:
            import orbax.checkpoint as ocp

            ckpt_dir = str(_resolve(ckpt_dir))
            ckptr = ocp.CheckpointManager(
                ckpt_dir,
                options=ocp.CheckpointManagerOptions(
                    max_to_keep=int(tcfg.get("keep_checkpoints", 3)),
                    best_fn=lambda m: -m.get("val_loss", m.get("train_loss", 1e9)),
                    create=True),
            )
            latest = ckptr.latest_step()
            if latest is not None:
                restored = ckptr.restore(
                    latest, args=ocp.args.Composite(state=ocp.args.StandardRestore(state)))
                state = restored["state"]
                print(f"[train_jax] resumed from checkpoint step {latest}")
        except ImportError:
            print("[train_jax] orbax not installed — checkpointing disabled "
                  "(pip install orbax-checkpoint).")

    def save_ckpt(step, metrics):
        # Save the full TrainState (for exact resume, incl. optimizer momentum) AND a
        # params-only item, so inference/eval can restore weights without having to
        # reconstruct the exact optimizer pytree — see adhan_slm.inference.load_model.
        if ckptr is None:
            return
        import orbax.checkpoint as ocp
        ckptr.save(step, args=ocp.args.Composite(
            state=ocp.args.StandardSave(state),
            params=ocp.args.StandardSave(state.params)), metrics=metrics)

    eval_every = int(tcfg.get("eval_every", max(1, steps // 10)))
    ckpt_every = int(tcfg.get("checkpoint_every", eval_every))

    run_params = {**{f"model.{k}": v for k, v in vars(model_cfg).items()},
                  "train.batch_size": batch, "train.seq_len": seq_len,
                  "train.learning_rate": lr, "train.max_steps": steps,
                  "params_millions": round(model_cfg.approx_params() / 1e6, 2)}

    # Pulling a JAX array to host (float(), .item(), etc.) blocks until that
    # step's computation finishes — doing it every iteration serializes step
    # N+1's dispatch behind step N's completion and throws away XLA's async
    # dispatch pipelining, which is where most single-GPU throughput lives in a
    # loop this small. Buffer `log_every` steps of device-side loss values and
    # do one batched host sync, instead of one sync per step.
    log_every = max(1, int(tcfg.get("log_every", 10)))
    tokens_per_step = batch * seq_len

    with track_run(experiment=cfg.get("experiment", "adhan-slm"),
                   run_name=cfg.get("run_name", "smoke" if smoke else None),
                   params=run_params,
                   data_version=cfg.get("data", {}).get("version"),
                   tracking_uri=cfg.get("mlflow_uri")) as run:
        pending_losses = []
        pending_steps = []
        window_start = time.perf_counter()

        def _flush(final_step):
            if not pending_losses:
                return
            # one host sync for the whole buffered window, not one per step
            loss_vals = jax.device_get(pending_losses)
            elapsed = time.perf_counter() - window_start
            toks_per_sec = tokens_per_step * len(pending_losses) / max(elapsed, 1e-9)
            for s, lv in zip(pending_steps, loss_vals):
                lv = float(lv)
                run.log_metric("train_loss", lv, step=s)
                run.log_metric("perplexity", math.exp(min(lv, 20.0)), step=s)
                run.log_metric("learning_rate", float(sched(s)), step=s)
            run.log_metric("tokens_per_sec", toks_per_sec, step=final_step)
            last_lv = float(loss_vals[-1])
            print(f"step {final_step:6d}  loss {last_lv:.4f}  ppl {math.exp(min(last_lv, 20.0)):.2f}"
                  f"  tok/s {toks_per_sec:,.0f}")

        best_val = float("inf")
        last_train_loss = float("nan")
        for step, b in enumerate(batches):
            state, loss = train_step(state, jnp.asarray(b))
            pending_losses.append(loss)
            pending_steps.append(step)
            if (step + 1) % log_every == 0 or step == steps - 1:
                _flush(step)
                last_train_loss = float(pending_losses[-1])
                pending_losses, pending_steps = [], []
                window_start = time.perf_counter()

            is_last = step == steps - 1
            if val_ds is not None and ((step + 1) % eval_every == 0 or is_last):
                val_loss, val_ppl = validate(state.params)
                if val_loss is not None:
                    run.log_metric("val_loss", val_loss, step=step)
                    run.log_metric("val_perplexity", val_ppl, step=step)
                    marker = ""
                    if val_loss < best_val:
                        best_val, marker = val_loss, "  *best"
                    print(f"           val_loss {val_loss:.4f}  val_ppl {val_ppl:.2f}{marker}")

            if ckptr is not None and ((step + 1) % ckpt_every == 0 or is_last):
                metrics = {"train_loss": last_train_loss}
                if val_ds is not None:
                    vl, _ = validate(state.params)
                    if vl is not None:
                        metrics["val_loss"] = vl
                save_ckpt(step, metrics)

            if step + 1 >= steps:   # infinite loader — stop at max_steps
                break

        if ckptr is not None:
            ckptr.wait_until_finished()
        print("done." + (" (smoke)" if smoke else ""))


def main():
    ap = argparse.ArgumentParser(description="Adhan SLM JAX trainer")
    ap.add_argument("--config", required=True)
    ap.add_argument("--smoke", action="store_true",
                    help="train a few steps on synthetic data to verify the loop")
    args = ap.parse_args()
    train(args.config, smoke=args.smoke)


if __name__ == "__main__":
    main()

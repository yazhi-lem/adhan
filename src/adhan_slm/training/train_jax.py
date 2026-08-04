"""Adhan SLM pretraining loop in JAX/Flax with MLflow tracking.

A runnable, jit-ed causal-LM training step wired to MLflow. Three modes:

    # 1. smoke: a few steps on synthetic data, proves the loop end-to-end
    python -m adhan_slm.training.train_jax \
        --config src/adhan_slm/configs/adhan_slm_tiny.yaml --smoke

    # 2. overfit-a-batch: repeat ONE real batch until the loss collapses.
    #    The standard "is my model/optimizer/data wiring actually correct?" test
    #    (roadmap Phase 3) — a model that cannot memorise one batch has a bug.
    python -m adhan_slm.training.train_jax \
        --config src/adhan_slm/configs/adhan_slm_nano_cpu.yaml --overfit-batch

    # 3. real run: streams packed swaram-tokenized shards, validates, checkpoints
    python -m adhan_slm.training.train_jax \
        --config src/adhan_slm/configs/adhan_slm_nano_cpu.yaml

CPU training is a first-class path (see docs/CPU_TRAINING.md): `--device cpu`
pins the backend, `train.grad_accum_steps` buys a large effective batch out of a
CPU-sized micro-batch, and the pre-flight banner estimates wall-clock before the
run commits.
"""

from __future__ import annotations

import argparse
import functools
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # src/ on path

from adhan_slm.core.exceptions import ConfigError, TrainingError  # noqa: E402
from adhan_slm.core.logging import get_logger  # noqa: E402
from adhan_slm.training.device import (  # noqa: E402
    configure_backend,
    describe_backend,
    log_preflight,
    resolve_dtype,
)
from adhan_slm.training.mlflow_utils import track_run  # noqa: E402

logger = get_logger(__name__)

# NOTE: nothing at module scope may import jax (directly or via `adhan_slm.model`,
# whose transformer module imports it eagerly). `main()` calls configure_backend()
# to set JAX_PLATFORMS / XLA_FLAGS, and XLA reads those exactly once — when the
# backend first initialises. An eager `from adhan_slm.model import AdhanConfig`
# here would initialise it during module import, i.e. before main() runs, and
# silently make `--device cpu` a no-op. Hence the lazy imports below.


def load_config(path: str) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def _config_from_dict(d: dict):
    from adhan_slm.model import AdhanConfig  # lazy: see module-level NOTE

    m = d.get("model", {})
    size = m.get("size")
    vocab = m.get("vocab_size", 8000)
    if size in ("nano", "tiny", "mini"):
        cfg = getattr(AdhanConfig, size)(vocab_size=vocab)
        # Size presets set the architecture; per-run overrides (dtype, context
        # length, boundary embeddings) still come from the YAML.
        for key in ("dtype", "max_seq_len", "use_boundary_emb", "dropout", "rope_theta"):
            if key in m:
                setattr(cfg, key, m[key])
        return cfg
    return AdhanConfig(**{k: v for k, v in m.items() if k != "size"})


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
    yields ``(batch, seq_len)`` int32 batches.
    """
    from adhan_slm.data import PackedDataset
    from adhan_slm.data.packing import load_manifest

    shards = cfg.get("data", {}).get("shards")
    if not shards:
        raise ConfigError("config data.shards is required for a non-smoke run")
    shard_dir = _resolve(shards)
    train_bin = shard_dir / "train.bin" if shard_dir.is_dir() else shard_dir
    if not train_bin.exists():
        raise FileNotFoundError(
            f"packed shard not found: {train_bin}. Build it first:\n"
            f"  python scripts/prepare_slm_corpus.py --corpus <path> --out {shard_dir}"
        )
    manifest = load_manifest(train_bin)
    logger.info(
        "shard %s: %s seqs x %d tok (%s tokens, %s)",
        train_bin.name,
        f"{manifest.n_sequences:,}",
        manifest.seq_len,
        f"{manifest.n_tokens:,}",
        manifest.dtype,
    )
    return PackedDataset.from_shard(
        train_bin, batch_size, manifest=manifest, seed=seed, infinite=infinite
    )


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
    return PackedDataset.from_shard(
        val_bin,
        batch_size,
        manifest=manifest,
        shuffle=False,
        infinite=False,
        drop_last=False,
    )


def _run_artifacts(config_path: str, cfg: dict):
    """Files worth attaching to the MLflow run: the config, and the corpus datasheet."""
    paths = [Path(config_path)]
    shards = cfg.get("data", {}).get("shards")
    if shards:
        datasheet = _resolve(shards) / "datasheet.json"
        if datasheet.exists():
            paths.append(datasheet)
    return [p for p in paths if p.exists()]


def _repeat_one_batch(source):
    """Yield the first batch of `source` forever — the overfit-a-batch sanity mode."""
    first = next(iter(source))
    while True:
        yield first


def train(
    config_path: str,
    smoke: bool = False,
    overfit_batch: bool = False,
    overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run training. Returns a summary dict (used by the integration tests).

    Args:
        config_path: YAML config (see ``src/adhan_slm/configs/``).
        smoke: train a few steps on synthetic data, no corpus/checkpoints needed.
        overfit_batch: repeat a single batch; loss must collapse toward 0.
        overrides: CLI overrides for ``train.*`` keys (``max_steps``, ``batch_size``,
            ``learning_rate``, ``grad_accum_steps``, ``seq_len``).
    """
    cfg = load_config(config_path)
    tcfg = dict(cfg.get("train", {}))
    tcfg.update({k: v for k, v in (overrides or {}).items() if v is not None})
    model_cfg = _config_from_dict(cfg)

    try:
        import jax
        import jax.numpy as jnp
        import optax
        from flax.training import train_state

        from adhan_slm.model import AdhanSLM
    except ImportError as e:
        logger.error("JAX stack not available (%s). Install: pip install -e '.[jax]'", e)
        logger.info("would train %s (~%.1fM params)", model_cfg, model_cfg.approx_params() / 1e6)
        return {"status": "skipped", "reason": f"jax unavailable: {e}"}

    backend = describe_backend()
    model_cfg.dtype = resolve_dtype(model_cfg.dtype, backend.platform)
    backend.compute_dtype = model_cfg.dtype

    # `max_steps` counts OPTIMIZER UPDATES. With grad_accum_steps=k the loop runs
    # k micro-batches per update, so the effective batch is batch_size*k while peak
    # memory stays at batch_size — the lever that makes a CPU run use a sane batch.
    accum = max(1, int(tcfg.get("grad_accum_steps", 1)))
    if smoke or overfit_batch:
        accum = 1  # sanity modes want one update per batch, not a smoothed average
    updates = 20 if smoke else int(tcfg.get("max_steps", 100000))
    if overfit_batch:
        # A sanity gate, not a run: a few hundred updates on one batch is enough for
        # the loss to collapse, whatever the config's max_steps says.
        updates = int((overrides or {}).get("max_steps") or 200)
    batch = 4 if smoke else int(tcfg.get("batch_size", 32))
    lr = float(tcfg.get("learning_rate", 3e-4))
    warmup = int(tcfg.get("warmup_steps", 2000))
    seed = int(tcfg.get("seed", 0))

    # For real runs the sequence length is fixed by the packed shard; smoke uses a
    # short synthetic length. Build the data source before init so dummy shapes match.
    val_ds = None
    if smoke:
        # Clamp to the model's context: a nano/CPU config with max_seq_len 64 must not
        # be smoke-tested at 128 tokens, or the check below rejects its own default.
        seq_len = int(tcfg.get("seq_len", min(128, model_cfg.max_seq_len)))
        batches = synthetic_batches(model_cfg.vocab_size, batch, seq_len, updates)
    else:
        # The model's vocab MUST equal the frozen tokenizer's size, or the embedding
        # rows won't line up with the ids in the shards. The trained merge count rarely
        # hits the configured target exactly, so trust vocab.json over the YAML.
        frozen = _frozen_vocab_size(cfg)
        if frozen is not None and frozen != model_cfg.vocab_size:
            logger.info(
                "vocab_size %d (config) -> %d (frozen tokenizer) to match shards",
                model_cfg.vocab_size,
                frozen,
            )
            model_cfg.vocab_size = frozen
        train_ds = data_iterator(cfg, batch, seed=seed, infinite=True)
        seq_len = train_ds.seq_len
        batches = _repeat_one_batch(train_ds) if overfit_batch else iter(train_ds)
        if not overfit_batch:
            val_ds = _val_iterator(cfg, batch)

    if seq_len > model_cfg.max_seq_len:
        raise ConfigError(
            f"packed shard sequence length {seq_len} exceeds model.max_seq_len "
            f"{model_cfg.max_seq_len}. Either raise model.max_seq_len or repack the "
            f"corpus with --seq-len {model_cfg.max_seq_len}."
        )

    model = AdhanSLM(model_cfg)
    key = jax.random.PRNGKey(seed)
    dummy = jnp.ones((batch, seq_len), dtype=jnp.int32)
    params = model.init(key, dummy)["params"]

    # warmup_steps must be strictly less than decay_steps or optax's internal
    # cosine phase length (decay_steps - warmup_steps) hits zero and raises —
    # bites `--smoke` runs where warmup_steps (config default 2000) otherwise
    # clamps equal to the tiny smoke step count. In the sanity modes the run is only
    # a couple hundred updates long, so a production warmup (200–2000) would eat the
    # whole run at a near-zero LR and make the overfit gate fail for the one reason
    # it is not meant to detect: too little training.
    if smoke or overfit_batch:
        warmup = min(warmup, max(1, updates // 10))
    warmup_steps = min(warmup, max(1, updates - 1))
    sched = optax.warmup_cosine_decay_schedule(
        0.0, lr, warmup_steps=warmup_steps, decay_steps=updates, end_value=lr * 0.1
    )
    tx = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(sched, weight_decay=float(tcfg.get("weight_decay", 0.1))),
    )
    if accum > 1:
        # MultiSteps accumulates gradients over k micro-batches and only then applies
        # one update, so the LR schedule and Adam moments advance per *update*.
        tx = optax.MultiSteps(tx, every_k_schedule=accum)
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
    completed = 0  # updates already done in a previous run, restored below
    if ckpt_dir and not smoke and not overfit_batch:
        try:
            import orbax.checkpoint as ocp

            ckpt_dir = str(_resolve(ckpt_dir))
            ckptr = ocp.CheckpointManager(
                ckpt_dir,
                options=ocp.CheckpointManagerOptions(
                    max_to_keep=int(tcfg.get("keep_checkpoints", 3)),
                    best_fn=lambda m: -m.get("val_loss", m.get("train_loss", 1e9)),
                    create=True,
                ),
            )
            latest = ckptr.latest_step()
            if latest is not None:
                restored = ckptr.restore(
                    latest, args=ocp.args.Composite(state=ocp.args.StandardRestore(state))
                )
                state = restored["state"]
                # Resume must continue the *global* step budget, not restart it.
                # Without this the loop's own counter went back to 0, so a run
                # resumed at step 3999/4000 would train another full 4000 updates
                # (and re-log metrics over the earlier steps). The optimizer's own
                # count lives in opt_state and is restored with it, so the LR
                # schedule was already correct — only the loop's bookkeeping wasn't.
                completed = latest + 1
                logger.info(
                    "resumed from checkpoint step %d (%d of %d updates already done)",
                    latest,
                    completed,
                    updates,
                )
        except ImportError:
            logger.warning(
                "orbax not installed — checkpointing disabled (pip install orbax-checkpoint)"
            )

    # Self-describing checkpoint: the architecture (AdhanConfig) and the frozen
    # tokenizer dir it was trained against, carried as a JSON item inside the
    # checkpoint itself instead of a separately-passed --config/--tokenizer-dir.
    # adhan_slm.inference.load_model reads this back so a checkpoint dir alone is
    # enough to reconstruct and restore the model — see docs/ARCHITECTURE_SWARAM_SLM.md.
    _shards = cfg.get("data", {}).get("shards")
    ckpt_metadata = {
        "model_config": {k: v for k, v in vars(model_cfg).items()},
        "tokenizer_dir": str(_resolve(_shards)) if _shards else None,
    }

    def save_ckpt(step, metrics):
        # Save the full TrainState (for exact resume, incl. optimizer momentum) AND a
        # params-only item, so inference/eval can restore weights without having to
        # reconstruct the exact optimizer pytree — see adhan_slm.inference.load_model.
        if ckptr is None:
            return
        import orbax.checkpoint as ocp

        ckptr.save(
            step,
            args=ocp.args.Composite(
                state=ocp.args.StandardSave(state),
                params=ocp.args.StandardSave(state.params),
                metadata=ocp.args.JsonSave(ckpt_metadata),
            ),
            metrics=metrics,
        )

    eval_every = int(tcfg.get("eval_every", max(1, updates // 10)))
    ckpt_every = int(tcfg.get("checkpoint_every", eval_every))
    log_every = max(1, int(tcfg.get("log_every", 10)))
    tokens_per_update = batch * seq_len * accum

    mode = "smoke" if smoke else ("overfit-batch" if overfit_batch else "pretrain")
    run_params = {
        **{f"model.{k}": v for k, v in vars(model_cfg).items()},
        "train.batch_size": batch,
        "train.grad_accum_steps": accum,
        "train.effective_batch": batch * accum,
        "train.seq_len": seq_len,
        "train.learning_rate": lr,
        "train.max_steps": updates,
        "train.resumed_from_step": completed,
        "train.mode": mode,
        "params_millions": round(model_cfg.approx_params() / 1e6, 2),
        **backend.as_params(),
    }
    eta_seconds = log_preflight(
        backend, tokens_per_update, updates, model_cfg.approx_params() / 1e6
    )

    result: Dict[str, Any] = {
        "status": "ok",
        "mode": mode,
        "backend": backend.platform,
        "compute_dtype": model_cfg.dtype,
        "updates": updates,
        "effective_batch": batch * accum,
        "seq_len": seq_len,
    }

    with track_run(
        experiment=cfg.get("experiment", "adhan-slm"),
        run_name=cfg.get("run_name", mode if (smoke or overfit_batch) else None),
        params=run_params,
        data_version=cfg.get("data", {}).get("version"),
        tracking_uri=cfg.get("mlflow_uri"),
    ) as run:
        run.set_tag("mode", mode)
        run.set_tag("backend", backend.platform)
        run.log_metric("eta_seconds_estimate", eta_seconds, step=0)
        # Reproducibility contract (roadmap §5): the exact config and the corpus
        # datasheet travel with the run, not just the flattened param table.
        for artifact in _run_artifacts(config_path, cfg):
            try:
                run.log_artifact(str(artifact), artifact_path="config")
            except Exception as exc:  # tracking must never take the run down
                logger.warning("could not log artifact %s to MLflow: %s", artifact, exc)

        # Pulling a JAX array to host (float(), .item(), etc.) blocks until that
        # step's computation finishes — doing it every iteration serializes step
        # N+1's dispatch behind step N's completion and throws away XLA's async
        # dispatch pipelining, which is where most throughput lives in a loop this
        # small. Buffer `log_every` updates of device-side loss values and do one
        # batched host sync, instead of one sync per step.
        pending_losses: list = []
        pending_steps: list = []
        window_start = time.perf_counter()
        first_loss = None
        last_train_loss = float("nan")
        last_toks_per_sec = float("nan")

        def _flush(final_step):
            nonlocal last_train_loss, last_toks_per_sec
            if not pending_losses:
                return
            # one host sync for the whole buffered window, not one per update
            loss_vals = jax.device_get(pending_losses)
            elapsed = time.perf_counter() - window_start
            toks_per_sec = tokens_per_update * len(pending_losses) / max(elapsed, 1e-9)
            for s, lv in zip(pending_steps, loss_vals):
                lv = float(lv)
                run.log_metric("train_loss", lv, step=s)
                run.log_metric("perplexity", math.exp(min(lv, 20.0)), step=s)
                run.log_metric("learning_rate", float(sched(s)), step=s)
            run.log_metric("tokens_per_sec", toks_per_sec, step=final_step)
            last_train_loss = float(loss_vals[-1])
            last_toks_per_sec = toks_per_sec
            logger.info(
                "step %6d  loss %.4f  ppl %.2f  tok/s %s",
                final_step,
                last_train_loss,
                math.exp(min(last_train_loss, 20.0)),
                f"{toks_per_sec:,.0f}",
            )

        best_val = float("inf")
        micro = 0
        step = completed - 1
        if completed >= updates:
            logger.info(
                "checkpoint is already at %d/%d updates — nothing left to train. "
                "Raise train.max_steps to continue.",
                completed,
                updates,
            )
        for b in batches:
            if completed >= updates:
                break
            state, loss = train_step(state, jnp.asarray(b))
            micro += 1
            if micro % accum:
                continue  # gradient still accumulating; no update happened yet
            step = completed + micro // accum - 1
            if first_loss is None:
                first_loss = loss
            pending_losses.append(loss)
            pending_steps.append(step)
            is_last = step == updates - 1
            if (step + 1) % log_every == 0 or is_last:
                _flush(step)
                pending_losses, pending_steps = [], []
                window_start = time.perf_counter()

            if val_ds is not None and ((step + 1) % eval_every == 0 or is_last):
                val_loss, val_ppl = validate(state.params)
                if val_loss is not None:
                    run.log_metric("val_loss", val_loss, step=step)
                    run.log_metric("val_perplexity", val_ppl, step=step)
                    marker = ""
                    if val_loss < best_val:
                        best_val, marker = val_loss, "  *best"
                    logger.info(
                        "           val_loss %.4f  val_ppl %.2f%s", val_loss, val_ppl, marker
                    )

            if ckptr is not None and ((step + 1) % ckpt_every == 0 or is_last):
                metrics = {"train_loss": last_train_loss}
                if val_ds is not None:
                    vl, _ = validate(state.params)
                    if vl is not None:
                        metrics["val_loss"] = vl
                save_ckpt(step, metrics)

            if step + 1 >= updates:  # infinite loader — stop at max_steps
                break

        if pending_losses:
            _flush(step)
        if ckptr is not None:
            ckptr.wait_until_finished()

        result.update(
            final_train_loss=last_train_loss,
            initial_train_loss=float(first_loss) if first_loss is not None else float("nan"),
            best_val_loss=None if best_val == float("inf") else best_val,
            tokens_per_sec=last_toks_per_sec,
            resumed_from_step=completed,
            updates_this_run=micro // accum,
        )
        if overfit_batch:
            _report_overfit(result)
        logger.info("done (%s).", mode)
    return result


def _report_overfit(result: Dict[str, Any], threshold: float = 0.5) -> None:
    """Judge the overfit-a-batch sanity run: loss on one batch must collapse.

    A correctly wired causal LM memorises a single batch almost completely, so the
    loss should fall far below its ``ln(vocab)`` starting point. If it does not, the
    bug is in the model / optimizer / data wiring, not in the hyperparameters — which
    is exactly why the roadmap makes this the gate before any long run.
    """
    initial = result.get("initial_train_loss", float("nan"))
    final = result.get("final_train_loss", float("nan"))
    ratio = final / initial if initial and initial == initial else float("nan")
    result["overfit_loss_ratio"] = ratio
    result["overfit_passed"] = bool(ratio == ratio and ratio < threshold)
    if result["overfit_passed"]:
        logger.info(
            "overfit-a-batch PASSED: loss %.4f -> %.4f (%.0f%% of start, target <%.0f%%)",
            initial,
            final,
            ratio * 100,
            threshold * 100,
        )
    else:
        logger.error(
            "overfit-a-batch FAILED: loss %.4f -> %.4f (%.0f%% of start, need <%.0f%%). "
            "The model could not memorise a single batch — suspect the model, the "
            "optimizer wiring or the token ids, not the learning rate.",
            initial,
            final,
            ratio * 100,
            threshold * 100,
        )


def main():
    ap = argparse.ArgumentParser(description="Adhan SLM JAX trainer")
    ap.add_argument("--config", required=True)
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="train a few steps on synthetic data to verify the loop",
    )
    ap.add_argument(
        "--overfit-batch",
        action="store_true",
        help="repeat one real batch until the loss collapses (model/wiring sanity gate)",
    )
    ap.add_argument(
        "--device",
        choices=["auto", "cpu", "gpu", "tpu"],
        default="auto",
        help="force the JAX backend (default: auto-detect)",
    )
    ap.add_argument(
        "--cpu-devices",
        type=int,
        default=None,
        help="expose N addressable CPU devices (for testing sharded code paths only)",
    )
    ap.add_argument("--max-steps", type=int, default=None, help="override train.max_steps")
    ap.add_argument("--batch-size", type=int, default=None, help="override train.batch_size")
    ap.add_argument(
        "--grad-accum-steps",
        type=int,
        default=None,
        help="override train.grad_accum_steps (effective batch = batch_size x this)",
    )
    ap.add_argument(
        "--learning-rate", type=float, default=None, help="override train.learning_rate"
    )
    args = ap.parse_args()

    if args.smoke and args.overfit_batch:
        raise ConfigError("--smoke and --overfit-batch are mutually exclusive")

    # Must happen before the trainer imports jax: XLA reads these once, at backend init.
    configure_backend(args.device, args.cpu_devices)

    result = train(
        args.config,
        smoke=args.smoke,
        overfit_batch=args.overfit_batch,
        overrides={
            "max_steps": args.max_steps,
            "batch_size": args.batch_size,
            "grad_accum_steps": args.grad_accum_steps,
            "learning_rate": args.learning_rate,
        },
    )
    if args.overfit_batch and result.get("overfit_passed") is False:
        raise TrainingError("overfit-a-batch sanity check failed — see log above")


if __name__ == "__main__":
    main()

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
import sys
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


def data_iterator(cfg: dict, model_cfg: AdhanConfig):
    """Real runs: stream packed swaram-tokenized shards here (Phase 2/3).

    TODO: replace synthetic batches with Grain/tf.data over sharded TFRecord/Arrow
    produced by the tokenizer, with sequence packing to max_seq_len.
    """
    raise NotImplementedError(
        "Wire a Grain/tf.data iterator over tokenized shards (roadmap Phase 2/3).")


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
    seq_len = 128 if smoke else model_cfg.max_seq_len
    lr = float(tcfg.get("learning_rate", 3e-4))
    warmup = int(tcfg.get("warmup_steps", 2000))

    model = AdhanSLM(model_cfg)
    key = jax.random.PRNGKey(int(tcfg.get("seed", 0)))
    dummy = jnp.ones((batch, seq_len), dtype=jnp.int32)
    params = model.init(key, dummy)["params"]

    sched = optax.warmup_cosine_decay_schedule(
        0.0, lr, warmup_steps=min(warmup, steps), decay_steps=steps, end_value=lr * 0.1)
    tx = optax.chain(optax.clip_by_global_norm(1.0),
                     optax.adamw(sched, weight_decay=float(tcfg.get("weight_decay", 0.1))))
    state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

    def loss_fn(params, batch):
        logits = model.apply({"params": params}, batch[:, :-1])
        targets = batch[:, 1:]
        ll = optax.softmax_cross_entropy_with_integer_labels(logits, targets)
        return ll.mean()

    @jax.jit
    def train_step(state, batch):
        loss, grads = jax.value_and_grad(loss_fn)(state.params, batch)
        return state.apply_gradients(grads=grads), loss

    if smoke:
        batches = synthetic_batches(model_cfg.vocab_size, batch, seq_len, steps)
    else:
        batches = data_iterator(cfg, model_cfg)  # raises until Phase 2/3 wired

    run_params = {**{f"model.{k}": v for k, v in vars(model_cfg).items()},
                  "train.batch_size": batch, "train.seq_len": seq_len,
                  "train.learning_rate": lr, "train.max_steps": steps,
                  "params_millions": round(model_cfg.approx_params() / 1e6, 2)}

    with track_run(experiment=cfg.get("experiment", "adhan-slm"),
                   run_name=cfg.get("run_name", "smoke" if smoke else None),
                   params=run_params,
                   data_version=cfg.get("data", {}).get("version"),
                   tracking_uri=cfg.get("mlflow_uri")) as run:
        for step, b in enumerate(batches):
            state, loss = train_step(state, jnp.asarray(b))
            lv = float(loss)
            run.log_metric("train_loss", lv, step=step)
            run.log_metric("perplexity", float(jnp.exp(loss)), step=step)
            run.log_metric("learning_rate", float(sched(step)), step=step)
            if step % max(1, steps // 10) == 0 or step == steps - 1:
                print(f"step {step:6d}  loss {lv:.4f}  ppl {float(jnp.exp(loss)):.2f}")
            # TODO(Phase 3): periodic eval + Orbax async checkpoint + best-by-val save
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

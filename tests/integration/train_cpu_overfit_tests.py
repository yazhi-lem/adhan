"""CPU training: the overfit-a-batch sanity gate (roadmap Phase 3).

A correctly wired causal LM memorises a single batch almost completely. If the loss
refuses to collapse, the bug is in the model, the optimizer wiring or the token ids —
and no amount of hyperparameter tuning will save the long run. This is the gate that
runs before any GPU hour is spent.
"""

from __future__ import annotations

import pytest

pytest.importorskip("jax", reason="CPU training tests need the JAX stack")
pytest.importorskip("flax", reason="CPU training tests need flax")
pytest.importorskip("optax", reason="CPU training tests need optax")

from adhan_slm.training import train_jax  # noqa: E402

pytestmark = [pytest.mark.integration, pytest.mark.jax, pytest.mark.slow]


@pytest.fixture(scope="module")
def overfit_result(cpu_config):
    """Run the gate once and share the result across this module's assertions."""
    cfg = cpu_config(name="overfit")
    return train_jax.train(str(cfg), overfit_batch=True, overrides={"max_steps": 120})


def test_gate_runs_in_overfit_mode(overfit_result):
    assert overfit_result["mode"] == "overfit-batch"


def test_loss_falls_over_the_run(overfit_result):
    assert overfit_result["initial_train_loss"] > overfit_result["final_train_loss"]


def test_gate_passes_on_a_correctly_wired_model(overfit_result):
    assert overfit_result["overfit_passed"], (
        "model failed to memorise a single batch "
        f"({overfit_result['initial_train_loss']:.3f} -> "
        f"{overfit_result['final_train_loss']:.3f}); the model/optimizer/data wiring is "
        "broken, not the hyperparameters"
    )


def test_gate_reports_the_loss_ratio_it_judged_on(overfit_result):
    """The ratio is the gate's verdict; it belongs in the result so CI can log it."""
    ratio = overfit_result["overfit_loss_ratio"]
    assert 0.0 <= ratio < 0.5, f"expected the loss to collapse below 50%, got {ratio:.3f}"


def test_gate_does_not_write_checkpoints(cpu_config, tmp_path):
    """A sanity gate must not leave a memorised-one-batch checkpoint behind for a real
    run to resume from."""
    ckpt = tmp_path / "gate_ckpt"
    cfg = cpu_config(name="overfit_nockpt", checkpoint_dir=str(ckpt))
    train_jax.train(str(cfg), overfit_batch=True, overrides={"max_steps": 20})

    assert not ckpt.exists() or not any(ckpt.iterdir()), "overfit mode must not checkpoint"

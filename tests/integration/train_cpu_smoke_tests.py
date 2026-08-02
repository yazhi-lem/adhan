"""CPU training: backend sanity + the zero-setup `--smoke` path.

`--smoke` trains a few steps on synthetic tokens and needs no corpus at all — the
fastest "does the loop run on this machine" check, and the first thing to try when
CPU training misbehaves.
"""

from __future__ import annotations

import pytest

jax = pytest.importorskip("jax", reason="CPU training tests need the JAX stack")
pytest.importorskip("flax", reason="CPU training tests need flax")
pytest.importorskip("optax", reason="CPU training tests need optax")

from adhan_slm.training import train_jax  # noqa: E402

pytestmark = [pytest.mark.integration, pytest.mark.jax]


def test_backend_is_cpu():
    """The rest of this suite is meaningless if something quietly grabbed a GPU."""
    assert jax.default_backend() == "cpu", f"expected CPU backend, got {jax.default_backend()}"


@pytest.fixture(scope="module")
def smoke_result(cpu_config):
    """One smoke run, shared across this module's assertions.

    Each `train()` call pays ~8s of XLA compilation for its graph shape, so runs are
    shared wherever the assertions are about the same run.
    """
    return train_jax.train(str(cpu_config(name="smoke")), smoke=True)


def test_smoke_run_needs_no_corpus(smoke_result):
    assert smoke_result["status"] == "ok"
    assert smoke_result["mode"] == "smoke"
    assert smoke_result["backend"] == "cpu"


def test_smoke_run_produces_a_real_loss(smoke_result):
    loss = smoke_result["final_train_loss"]
    assert loss == loss, "loss must not be NaN"
    assert loss > 0


def test_smoke_run_clamps_seq_len_to_model_context(cpu_config):
    """The synthetic length must respect `model.max_seq_len` — a nano/CPU config with a
    short context used to be rejected by the trainer's own context check."""
    cfg = cpu_config(name="smoke_short", model={"max_seq_len": 32})
    result = train_jax.train(str(cfg), smoke=True)

    assert result["status"] == "ok"
    assert result["seq_len"] <= 32


def test_smoke_run_reports_bfloat16_compute_dtype(smoke_result):
    """bf16 is the measured-faster CPU default; a silent downgrade would hide a 1.5x."""
    assert smoke_result["compute_dtype"] == "bfloat16"

"""CPU training: Orbax checkpointing and resume.

Long CPU runs get interrupted, so resume is the expected path rather than an edge
case. The contract under test: a restart continues the *global* step budget. A resume
that silently re-ran `max_steps` from scratch would double the wall clock and corrupt
the metric history.
"""

from __future__ import annotations

import pytest
import yaml

pytest.importorskip("jax", reason="CPU training tests need the JAX stack")
pytest.importorskip("flax", reason="CPU training tests need flax")
pytest.importorskip("optax", reason="CPU training tests need optax")
pytest.importorskip("orbax.checkpoint", reason="resume tests need orbax-checkpoint")

from adhan_slm.training import train_jax  # noqa: E402

pytestmark = [pytest.mark.integration, pytest.mark.jax, pytest.mark.slow]


def _with_max_steps(config_path, max_steps: int, out_path):
    """Copy a config with a new `train.max_steps` — a resume with more budget."""
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    cfg["train"]["max_steps"] = max_steps
    out_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return out_path


@pytest.fixture(scope="module")
def resumed(cpu_config, tmp_path_factory):
    """Train 10 updates, checkpoint, then resume with a 16-update budget."""
    ckpt = tmp_path_factory.mktemp("resume_ckpt")
    first = cpu_config(
        name="resume_a",
        checkpoint_dir=str(ckpt),
        max_steps=10,
        checkpoint_every=5,
        eval_every=10,
    )
    initial = train_jax.train(str(first))

    second = _with_max_steps(first, 16, first.parent / "resume_b.yaml")
    return initial, train_jax.train(str(second)), ckpt


def test_first_run_completes_and_checkpoints(resumed):
    initial, _, ckpt = resumed
    assert initial["status"] == "ok"
    assert any(ckpt.iterdir()), "no checkpoint was written"


def test_resume_reports_the_step_it_restarted_from(resumed):
    _, second, _ = resumed
    assert second["status"] == "ok"
    assert second["resumed_from_step"] == 10


def test_resume_runs_only_the_remaining_updates(resumed):
    """10 of 16 updates were already done, so 6 remain — not another full 16."""
    _, second, _ = resumed
    assert second["updates_this_run"] == 6


def test_resume_with_no_budget_left_trains_nothing(cpu_config, tmp_path_factory):
    """Re-running a finished config must be a no-op, not a second full run."""
    ckpt = tmp_path_factory.mktemp("done_ckpt")
    cfg = cpu_config(
        name="resume_done",
        checkpoint_dir=str(ckpt),
        max_steps=6,
        checkpoint_every=3,
        eval_every=6,
    )
    assert train_jax.train(str(cfg))["status"] == "ok"

    again = train_jax.train(str(cfg))
    assert again["resumed_from_step"] == 6
    assert again["updates_this_run"] == 0

"""Tests for training debug diagnostics module."""

from __future__ import annotations

import pytest

from adhan_slm.training.debug_diagnostics import (
    StepDiagnostics,
    TrainingDiagnostics,
    _compute_update_ratios,
    _group_grad_norms,
)

pytestmark = pytest.mark.unit


def test_group_grad_norms_numpy():
    grads = {
        "block_0": {"qkv": [1.0, 2.0], "proj": [3.0]},
        "block_1": {"qkv": [0.0], "proj": [4.0, 5.0]},
        "tok_emb": [1.0, 1.0],
    }
    norms = _group_grad_norms(grads)
    # block_0: sqrt(1^2 + 2^2 + 3^2) = sqrt(14)
    assert pytest.approx(norms["block_0"]) == (14**0.5)
    # block_1: sqrt(0^2 + 4^2 + 5^2) = sqrt(41)
    assert pytest.approx(norms["block_1"]) == (41**0.5)
    # tok_emb (no dot in key) falls into "global" group
    assert pytest.approx(norms["global"]) == (2**0.5)


def test_compute_update_ratios():
    before = {"w": [1.0, 2.0], "b": [0.5]}
    after = {"w": [2.0, 4.0], "b": [0.5]}  # w doubled, b unchanged
    ratios = _compute_update_ratios(before, after)
    # w delta norm = ||[1, 2]|| = sqrt(5); before norm = sqrt(5); ratio = 1.0
    assert pytest.approx(ratios["w"]) == 1.0
    # b delta norm = 0; before_norm > 0 so ratio = 0.0 is recorded
    assert "b" in ratios
    assert pytest.approx(ratios["b"]) == 0.0


def test_step_diagnostics_defaults():
    diag = StepDiagnostics(step=5)
    assert diag.step == 5
    assert diag.loss == 0.0
    assert diag.grad_norms == {}


def test_training_diagnostics_detect_nans():
    diag = TrainingDiagnostics(model=None, log_every=1)
    # Simulate a logged step with NaN in activation stats
    from adhan_slm.training.debug_diagnostics import StepDiagnostics

    d = StepDiagnostics(step=1)
    d.activation_stats = {"layer_0": {"mean": 0.0, "std": float("nan")}}
    diag.history.append(d)
    nans = diag.detect_nans()
    assert len(nans) == 1
    assert "std" in nans[0]


def test_training_diagnostics_detect_spikes_no_history():
    diag = TrainingDiagnostics(model=None, log_every=1)
    spikes = diag.detect_spikes()
    assert spikes == []

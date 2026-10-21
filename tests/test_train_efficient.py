"""Tests for scripts/train_efficient.py (the PyTorch trainer).

Skipped when PyTorch is not installed.
"""

import argparse
import json
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import train_efficient as te  # noqa: E402

pytestmark = pytest.mark.pytorch


def test_build_tier_config_covers_every_size_with_given_vocab():
    for size in ("nano", "tiny", "mini"):
        cfg = te.build_tier_config(size, vocab_size=3000, seq_len=64)
        assert cfg["vocab_size"] == 3000
        assert cfg["max_seq_len"] == 64
        te.AdhanModel(**{**cfg, "n_layers": 1})  # constructs without error


def test_grad_scaler_only_for_fp16_on_cuda():
    assert te.needs_grad_scaler("cuda", torch.float16) is True
    assert te.needs_grad_scaler("cuda", torch.bfloat16) is False
    assert te.needs_grad_scaler("cpu", torch.bfloat16) is False
    assert te.needs_grad_scaler("cpu", torch.float16) is False


def test_overfit_gate_uses_frozen_vocab_not_cli_default(tmp_path, monkeypatch):
    """Frozen vocab (3000) is smaller than the CLI default (12000): the gate used to
    sample token ids up to 12000 and crash with an embedding IndexError."""
    (tmp_path / "vocab.json").write_text(json.dumps({f"t{i}": i for i in range(3000)}))
    monkeypatch.setattr(te, "configure_hardware", lambda: (torch.device("cpu"), "float32"))
    args = argparse.Namespace(
        data_dir=str(tmp_path),
        checkpoint_dir=str(tmp_path / "ckpt"),
        size="nano",
        vocab_size=12000,
        seq_len=8,
        batch_size=1,
        lr=1.5e-3,
        overfit_batch=True,
    )
    te.train(args)  # must not raise

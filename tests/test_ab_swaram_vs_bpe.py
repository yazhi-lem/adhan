"""Unit tests for the Swaram-vs-BPE A/B harness (N14)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "ab_swaram_vs_bpe.py"
_spec = importlib.util.spec_from_file_location("ab_swaram_vs_bpe", _SCRIPT)
ab = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ab)


def _m(name, tokens, aks):
    return {
        "tokenizer": name,
        "tokens": tokens,
        "aksharas": aks,
        "fertility_tok_per_akshara": tokens / aks,
    }


def test_verdict_supports_swaram_when_materially_more_compressed() -> None:
    v = ab._verdict(_m("swaram", 50, 100), _m("bpe", 100, 100))
    assert v["status"] == "swaram_supported"
    assert v["swaram_wins_by_10pct"] is True


def test_verdict_rejects_when_gain_is_below_ten_percent() -> None:
    v = ab._verdict(_m("swaram", 100, 100), _m("bpe", 105, 100))
    assert v["status"] == "thesis_not_supported_on_this_sample"
    assert v["swaram_wins_by_10pct"] is False


def test_held_out_split_is_deterministic_and_seed_sensitive() -> None:
    a = [ab.is_held_out(i, seed=0, val_frac=0.02) for i in range(200)]
    b = [ab.is_held_out(i, seed=0, val_frac=0.02) for i in range(200)]
    assert a == b
    # A different seed must not pick the identical set with overwhelming probability.
    c = [ab.is_held_out(i, seed=1, val_frac=0.02) for i in range(200)]
    assert a != c


def test_overall_verdict_keys_off_the_strongest_baseline() -> None:
    """A win against the weak baseline must not mask a loss against the strong one."""
    verdicts = {
        "byte": {"swaram_advantage_relative": 1.49},
        "metaspace": {"swaram_advantage_relative": -0.34},
    }
    overall = ab._overall_verdict(verdicts)
    assert overall["status"] == "thesis_not_supported_on_this_sample"
    assert overall["swaram_advantage_vs_strongest_baseline"] == -0.34

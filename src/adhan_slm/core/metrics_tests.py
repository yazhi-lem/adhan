"""Tests for throughput / latency / training metrics (`adhan_slm.core.metrics`).

Run: PYTHONPATH=src python -m adhan_slm.core.metrics_tests
"""

from __future__ import annotations

from .metrics import LatencyTracker, ThroughputTracker, TrainingMetrics
from .selftest import run_module_tests


def test_throughput_tracker_reports_a_positive_rate() -> None:
    tracker = ThroughputTracker(window_size=10)
    for _ in range(5):
        tracker.update(num_tokens=1024, num_examples=32)

    tps = tracker.tokens_per_second()
    assert tps is not None and tps > 0


def test_throughput_tracker_needs_two_samples() -> None:
    """One sample spans no interval, so a rate would be a divide-by-zero fiction."""
    tracker = ThroughputTracker(window_size=10)
    assert tracker.tokens_per_second() is None
    tracker.update(num_tokens=1024, num_examples=32)
    assert tracker.tokens_per_second() is None


def test_latency_tracker_reports_all_percentiles() -> None:
    tracker = LatencyTracker()
    for lat in (0.001, 0.002, 0.0015, 0.0025, 0.002):
        tracker.update(lat)

    assert tracker.mean() is not None
    assert tracker.min() is not None
    assert tracker.max() is not None
    assert tracker.p95() is not None
    assert tracker.p99() is not None


def test_training_metrics_summary_tracks_current_and_mean_loss() -> None:
    metrics = TrainingMetrics()
    metrics.update_loss(2.5)
    metrics.update_loss(2.3)
    metrics.update_eval_loss(2.4)
    metrics.throughput.update(1024, 32)

    summary = metrics.summary()
    assert "loss_mean" in summary
    assert "loss_current" in summary
    assert summary["loss_current"] == 2.3


if __name__ == "__main__":
    run_module_tests(globals(), "core metrics")

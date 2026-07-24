"""Performance metrics and monitoring for Adhan SLM.

Tracks throughput, latency, and resource utilization during training and inference.
"""

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Optional


@dataclass
class ThroughputTracker:
    """Tracks tokens per second and examples per second."""

    window_size: int = 100  # Number of batches to average over
    tokens_processed: Deque[int] = field(init=False)
    examples_processed: Deque[int] = field(init=False)
    timestamps: Deque[float] = field(init=False)

    def __post_init__(self) -> None:
        self.tokens_processed = deque(maxlen=self.window_size)
        self.examples_processed = deque(maxlen=self.window_size)
        self.timestamps = deque(maxlen=self.window_size)
    def update(self, num_tokens: int, num_examples: int) -> None:
        """Record tokens and examples processed.

        Args:
            num_tokens: Number of tokens processed
            num_examples: Number of examples processed
        """
        self.tokens_processed.append(num_tokens)
        self.examples_processed.append(num_examples)
        self.timestamps.append(time.time())

    def tokens_per_second(self) -> Optional[float]:
        """Calculate tokens per second over the window.

        Returns:
            Tokens per second, or None if not enough data
        """
        if len(self.tokens_processed) < 2:
            return None

        total_tokens = sum(self.tokens_processed)
        time_delta = self.timestamps[-1] - self.timestamps[0]

        if time_delta <= 0:
            return None

        return total_tokens / time_delta

    def examples_per_second(self) -> Optional[float]:
        """Calculate examples per second over the window.

        Returns:
            Examples per second, or None if not enough data
        """
        if len(self.examples_processed) < 2:
            return None

        total_examples = sum(self.examples_processed)
        time_delta = self.timestamps[-1] - self.timestamps[0]

        if time_delta <= 0:
            return None

        return total_examples / time_delta

    def reset(self) -> None:
        """Reset all counters."""
        self.tokens_processed.clear()
        self.examples_processed.clear()
        self.timestamps.clear()


@dataclass
class LatencyTracker:
    """Tracks latency statistics (min, max, mean, p95, p99)."""

    latencies: Deque[float] = field(default_factory=lambda: deque(maxlen=1000))
    window_size: int = 1000

    def update(self, latency: float) -> None:
        """Record a latency measurement.

        Args:
            latency: Latency in seconds
        """
        self.latencies.append(latency)

    def mean(self) -> Optional[float]:
        """Calculate mean latency."""
        if not self.latencies:
            return None
        return sum(self.latencies) / len(self.latencies)

    def min(self) -> Optional[float]:
        """Get minimum latency."""
        return min(self.latencies) if self.latencies else None

    def max(self) -> Optional[float]:
        """Get maximum latency."""
        return max(self.latencies) if self.latencies else None

    def percentile(self, p: float) -> Optional[float]:
        """Calculate percentile latency (p in [0, 100]).

        Args:
            p: Percentile (0-100)

        Returns:
            Latency at percentile, or None if not enough data
        """
        if not self.latencies:
            return None

        sorted_latencies = sorted(self.latencies)
        idx = int(len(sorted_latencies) * (p / 100))
        return sorted_latencies[min(idx, len(sorted_latencies) - 1)]

    def p95(self) -> Optional[float]:
        """Get 95th percentile latency."""
        return self.percentile(95)

    def p99(self) -> Optional[float]:
        """Get 99th percentile latency."""
        return self.percentile(99)

    def reset(self) -> None:
        """Reset all latencies."""
        self.latencies.clear()


@dataclass
class ResourceMonitor:
    """Monitors GPU/CPU memory and compute utilization."""

    peak_memory_bytes: int = 0
    current_memory_bytes: int = 0
    peak_gpu_memory_bytes: int = 0
    current_gpu_memory_bytes: int = 0

    def update_memory(self, current: int, peak: int) -> None:
        """Update CPU memory stats.

        Args:
            current: Current memory usage in bytes
            peak: Peak memory usage in bytes
        """
        self.current_memory_bytes = current
        self.peak_memory_bytes = max(self.peak_memory_bytes, peak)

    def update_gpu_memory(self, current: int, peak: int) -> None:
        """Update GPU memory stats.

        Args:
            current: Current GPU memory usage in bytes
            peak: Peak GPU memory usage in bytes
        """
        self.current_gpu_memory_bytes = current
        self.peak_gpu_memory_bytes = max(self.peak_gpu_memory_bytes, peak)

    def to_dict(self) -> Dict[str, float]:
        """Export stats as dictionary.

        Returns:
            Dictionary with memory stats
        """
        return {
            "peak_memory_mb": self.peak_memory_bytes / (1024 ** 2),
            "current_memory_mb": self.current_memory_bytes / (1024 ** 2),
            "peak_gpu_memory_mb": self.peak_gpu_memory_bytes / (1024 ** 2),
            "current_gpu_memory_mb": self.current_gpu_memory_bytes / (1024 ** 2),
        }


@dataclass
class TrainingMetrics:
    """Aggregates all training metrics.

    Tracks loss, perplexity, throughput, latency, and resource utilization.
    """

    throughput: ThroughputTracker = field(default_factory=ThroughputTracker)
    latency: LatencyTracker = field(default_factory=LatencyTracker)
    resources: ResourceMonitor = field(default_factory=ResourceMonitor)
    loss_history: Deque[float] = field(default_factory=lambda: deque(maxlen=1000))
    eval_loss_history: Deque[float] = field(default_factory=lambda: deque(maxlen=100))

    def update_loss(self, loss: float) -> None:
        """Record training loss.

        Args:
            loss: Training loss value
        """
        self.loss_history.append(loss)

    def update_eval_loss(self, eval_loss: float) -> None:
        """Record evaluation loss.

        Args:
            eval_loss: Evaluation loss value
        """
        self.eval_loss_history.append(eval_loss)

    def summary(self) -> Dict[str, float]:
        """Get summary statistics.

        Returns:
            Dictionary with all metrics
        """
        summary: Dict[str, float] = {}

        # Throughput metrics
        tps = self.throughput.tokens_per_second()
        if tps is not None:
            summary["tokens_per_second"] = tps

        eps = self.throughput.examples_per_second()
        if eps is not None:
            summary["examples_per_second"] = eps

        # Latency metrics
        if self.latency.latencies:
            summary["latency_mean_ms"] = (self.latency.mean() or 0) * 1000
            summary["latency_p95_ms"] = (self.latency.p95() or 0) * 1000
            summary["latency_p99_ms"] = (self.latency.p99() or 0) * 1000
            summary["latency_min_ms"] = (self.latency.min() or 0) * 1000
            summary["latency_max_ms"] = (self.latency.max() or 0) * 1000

        # Loss metrics
        if self.loss_history:
            summary["loss_mean"] = sum(self.loss_history) / len(self.loss_history)
            summary["loss_current"] = self.loss_history[-1]

        if self.eval_loss_history:
            summary["eval_loss_mean"] = sum(self.eval_loss_history) / len(
                self.eval_loss_history
            )
            summary["eval_loss_current"] = self.eval_loss_history[-1]

        # Resource metrics
        summary.update(self.resources.to_dict())

        return summary

    def reset(self) -> None:
        """Reset all metrics."""
        self.throughput.reset()
        self.latency.reset()
        self.loss_history.clear()
        self.eval_loss_history.clear()

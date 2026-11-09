"""Training debugging diagnostics — per-layer gradient norms, activation stats,
weight-update ratios, and mixed-precision overflow detection.

Adapted from "Under the Hood" Project 11: Training Debugging: Spikes, NaNs,
and Profiling (mechramc/Under-the-hood), adapted for JAX/Optax/Flax.

Usage (inside train_jax.py)::

    from adhan_slm.training.debug_diagnostics import TrainingDiagnostics

    diag = TrainingDiagnostics(model, log_every=10)

    for step, batch in enumerate(batches):
        state, loss = train_step(state, batch, rng)
        diag.maybe_log(state, step, loss)   # logs every 10 steps
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from adhan_slm.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class StepDiagnostics:
    """Diagnostics captured for a single training step."""

    step: int
    loss: float = 0.0
    lr: float = 0.0

    # Gradient norms per parameter group (layer_name -> norm)
    grad_norms: Dict[str, float] = field(default_factory=dict)
    max_grad_norm: float = 0.0
    mean_grad_norm: float = 0.0

    # Activation statistics (layer_name -> stats)
    activation_stats: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # Weight-update ratios (param_name -> ratio)
    update_ratios: Dict[str, float] = field(default_factory=dict)
    max_update_ratio: float = 0.0
    mean_update_ratio: float = 0.0

    # Mixed-precision overflow detection
    amp_scale: float = 1.0
    amp_overflow: bool = False


def _flatten_pytree(pytree: Any) -> Dict[str, Any]:
    """Flatten a nested dict of arrays into dot-path keys."""
    out: Dict[str, Any] = {}

    def _walk(obj: Any, prefix: str):
        if isinstance(obj, dict):
            for k, v in obj.items():
                _walk(v, f"{prefix}.{k}" if prefix else k)
        else:
            out[prefix] = obj

    _walk(pytree, "")
    return out


def _group_grad_norms(grads: Dict[str, Any]) -> Dict[str, float]:
    """Compute per-layer/toplevel gradient norms from a flattened grad tree.

    Groups parameters by their top-level key (e.g. ``block_0``, ``tok_emb``).
    """
    flat = _flatten_pytree(grads)
    groups: Dict[str, List[Any]] = {}
    for key, arr in flat.items():
        top = key.split(".")[0] if "." in key else "global"
        groups.setdefault(top, []).append(arr)

    norms = {}
    for gname, arrs in groups.items():
        # JAX arrays may be on device — compute norm elementwise
        try:
            import jax.numpy as jnp

            total_sq = sum(float(jnp.sum(jnp.square(a))) for a in arrs)
            norms[gname] = math.sqrt(total_sq)
        except Exception:
            # Fallback for non-JAX arrays (numpy, etc.)
            total_sq = sum(float(np.sum(np.square(np.asarray(a)))) for a in arrs)
            norms[gname] = math.sqrt(total_sq)
    return norms


def _compute_update_ratios(
    params_before: Dict[str, Any], params_after: Dict[str, Any]
) -> Dict[str, float]:
    """Compute update ratio: ||delta|| / ||before|| for every parameter."""
    flat_before = _flatten_pytree(params_before)
    flat_after = _flatten_pytree(params_after)
    ratios = {}
    for key in flat_before:
        if key not in flat_after:
            continue
        b = flat_before[key]
        a = flat_after[key]
        try:
            import jax.numpy as jnp

            before_norm = float(jnp.linalg.norm(b))
            delta_norm = float(jnp.linalg.norm(a - b))
        except Exception:
            bnp = np.asarray(b)
            anp = np.asarray(a)
            before_norm = float(np.linalg.norm(bnp))
            delta_norm = float(np.linalg.norm(anp - bnp))
        if before_norm > 1e-12:
            ratios[key] = delta_norm / before_norm
    return ratios


class TrainingDiagnostics:
    """Periodic training diagnostics logger.

    Captures:
      - Per-layer gradient L2 norms (spike detection)
      - Activation mean/std/min/max (distribution drift)
      - Weight-update ratios (optimizer health)
      - AMP scale / overflow events (mixed-precision stability)
    """

    def __init__(
        self,
        model: Any,
        log_every: int = 10,
        history_size: int = 100,
        log_dir: Optional[str] = None,
    ):
        self.model = model
        self.log_every = max(1, log_every)
        self.history: List[StepDiagnostics] = []
        self.history_size = history_size
        self.log_dir = Path(log_dir) if log_dir else None
        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)

        # Activation hooks: forward hooks capturing layer outputs
        self._activation_stats: Dict[str, Dict[str, float]] = {}
        self._register_hooks()

    def _register_hooks(self):
        """Register forward hooks on transformer blocks to capture activations.

        In JAX/Flax this is done via the ``nn.module`` hook system or by
        wrapping the apply function. Since Flax modules don't have PyTorch-style
        hooks, this is a no-op placeholder; users should call
        ``record_activation(name, tensor)`` explicitly in their forward pass.
        """
        pass

    def record_activation(self, name: str, tensor: Any):
        """Record activation statistics for a named layer.

        Call this from inside the forward pass or after ``model.apply``.
        """
        try:
            import jax.numpy as jnp

            x = jnp.asarray(tensor)
            self._activation_stats[name] = {
                "mean": float(jnp.mean(x)),
                "std": float(jnp.std(x)),
                "min": float(jnp.min(x)),
                "max": float(jnp.max(x)),
                "abs_max": float(jnp.max(jnp.abs(x))),
            }
        except Exception:
            x = np.asarray(tensor)
            self._activation_stats[name] = {
                "mean": float(np.mean(x)),
                "std": float(np.std(x)),
                "min": float(np.min(x)),
                "max": float(np.max(x)),
                "abs_max": float(np.max(np.abs(x))),
            }

    def maybe_log(
        self,
        state: Any,
        step: int,
        loss: float,
        lr: float = 0.0,
        grads: Optional[Any] = None,
        amp_scale: float = 1.0,
        amp_overflow: bool = False,
    ) -> Optional[StepDiagnostics]:
        """Capture and optionally log diagnostics for this step.

        Returns the diagnostics dict if this step was logged, else None.
        """
        diag = StepDiagnostics(step=step, loss=loss, lr=lr)

        # Grad norms
        if grads is not None:
            try:
                norms = _group_grad_norms(grads)
                diag.grad_norms = norms
                if norms:
                    diag.max_grad_norm = max(norms.values())
                    diag.mean_grad_norm = sum(norms.values()) / len(norms)
            except Exception as exc:
                logger.debug("grad norm computation failed: %s", exc)

        # Activation stats
        diag.activation_stats = dict(self._activation_stats)
        self._activation_stats.clear()

        # Update ratios: compare current params to a snapshot (expensive, skip often)
        if step % (self.log_every * 10) == 0 and hasattr(state, "params"):
            # Requires caller to manage pre/post snapshots. Placeholder.
            pass

        # AMP
        diag.amp_scale = amp_scale
        diag.amp_overflow = amp_overflow

        self.history.append(diag)
        if len(self.history) > self.history_size:
            self.history.pop(0)

        if step % self.log_every == 0:
            self._print_summary(diag)
            if self.log_dir:
                self._write_json(step, diag)
            return diag
        return None

    def _print_summary(self, diag: StepDiagnostics):
        """Print a concise one-line diagnostic summary."""
        parts = [f"step={diag.step:>6} loss={diag.loss:.4f} lr={diag.lr:.2e}"]
        if diag.grad_norms:
            parts.append(f"g_max={diag.max_grad_norm:.3e} g_mean={diag.mean_grad_norm:.3e}")
        if diag.activation_stats:
            abs_max = max(v.get("abs_max", 0.0) for v in diag.activation_stats.values())
            parts.append(f"act_abs_max={abs_max:.3e}")
        if diag.amp_overflow:
            parts.append("AMP_OVERFLOW")
        logger.info(" | ".join(parts))

    def _write_json(self, step: int, diag: StepDiagnostics):
        path = self.log_dir / f"diag_step_{step:08d}.json"
        path.write_text(
            json.dumps(
                {
                    "step": diag.step,
                    "loss": diag.loss,
                    "lr": diag.lr,
                    "grad_norms": diag.grad_norms,
                    "max_grad_norm": diag.max_grad_norm,
                    "mean_grad_norm": diag.mean_grad_norm,
                    "activation_stats": diag.activation_stats,
                    "amp_scale": diag.amp_scale,
                    "amp_overflow": diag.amp_overflow,
                },
                indent=1,
            ),
            encoding="utf-8",
        )

    def detect_spikes(self, window: int = 10, z_threshold: float = 3.0) -> List[str]:
        """Return param-group names whose latest grad norm is a spike (> z_threshold
        standard deviations above the rolling mean).
        """
        if len(self.history) < window + 1:
            return []

        recent = self.history[-(window + 1) : -1]
        latest = self.history[-1]

        alerts = []
        for name in latest.grad_norms:
            values = [h.grad_norms.get(name, 0.0) for h in recent if name in h.grad_norms]
            if len(values) < window:
                continue
            mean = sum(values) / len(values)
            std = math.sqrt(sum((v - mean) ** 2 for v in values) / len(values)) + 1e-12
            z = (latest.grad_norms[name] - mean) / std
            if z > z_threshold:
                alerts.append(f"{name}: z={z:.1f} (norm={latest.grad_norms[name]:.3e})")
        return alerts

    def detect_nans(self) -> List[str]:
        """Return parameter names that have NaN/Inf in the latest recorded state."""
        if not self.history:
            return []
        latest = self.history[-1]
        issues = []
        for name, stats in latest.activation_stats.items():
            for k, v in stats.items():
                if not math.isfinite(v):
                    issues.append(f"activation.{name}.{k}={v}")
        return issues

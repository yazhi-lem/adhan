"""Backend / precision resolution for Adhan SLM training runs.

Roadmap Phase 3 asks for "mixed precision (bf16, already wired) tuned on real data".
Tuning it means the loop must *know* what it is running on: the same config that is
fast on an A100 silently turns into a multi-day job on a laptop CPU, and a config
that asks for a compute dtype the backend cannot honour should say so up front
rather than after an hour of warmup.

This module is deliberately importable **without** JAX (so `--help`, config
validation and unit tests work in a minimal env) and does two jobs:

1. **Pre-import XLA configuration** (`configure_backend`). XLA reads ``XLA_FLAGS``
   and ``JAX_PLATFORMS`` from the environment *once*, when the jax backend is first
   initialised. So this has to run before ``import jax`` — hence a separate module
   called from ``main()`` before the trainer imports the JAX stack.
2. **Post-import introspection** (`describe_backend`, `resolve_dtype`,
   `estimate_runtime`) so every run logs its device, its resolved compute dtype and
   an honest wall-clock estimate to MLflow.

Measured on this repo's CI-class 4-core x86 CPU (jax 0.10, `adhan-nano`,
batch 8 × 256 tok): bfloat16 ≈ 3.0k tok/s, float32 ≈ 2.0k tok/s. bf16 is *not*
a pessimisation on CPU — XLA:CPU lowers it fine — so CPU runs keep bf16 by
default and only `float16` is rejected (no CPU kernels, silent upcast).
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from typing import Dict, Optional

from adhan_slm.core.exceptions import ConfigError
from adhan_slm.core.logging import get_logger

logger = get_logger(__name__)

#: Compute dtypes the Flax model is known to run correctly under, per platform.
_SUPPORTED_DTYPES = {
    "cpu": ("float32", "bfloat16"),
    "gpu": ("float32", "bfloat16", "float16"),
    "tpu": ("float32", "bfloat16"),
}

#: Rough steady-state throughput in tokens/sec, used only for the pre-flight ETA.
#: CPU number is measured (see module docstring); GPU/TPU are conservative
#: order-of-magnitude placeholders and get corrected by the live `tokens_per_sec`
#: metric a few steps into the run.
_TOKENS_PER_SEC_HINT = {"cpu": 3_000.0, "gpu": 150_000.0, "tpu": 400_000.0}


@dataclass
class BackendInfo:
    """What the training loop actually ended up running on."""

    platform: str  # "cpu" | "gpu" | "tpu"
    device_count: int
    device_kind: str
    host_cores: Optional[int]
    jax_version: str
    compute_dtype: str

    def as_params(self) -> Dict[str, object]:
        """Flatten for MLflow ``log_params`` (prefixed to group in the UI)."""
        return {f"runtime.{k}": v for k, v in asdict(self).items() if v is not None}


def configure_backend(device: Optional[str] = None, cpu_devices: Optional[int] = None) -> None:
    """Set XLA/JAX environment knobs. **Must be called before ``import jax``.**

    Args:
        device: force a platform — ``"cpu"``, ``"gpu"`` or ``"tpu"``. ``None`` (or
            ``"auto"``) leaves JAX's own accelerator discovery alone, so a GPU box
            keeps using its GPU.
        cpu_devices: expose N addressable CPU devices instead of 1. Only useful for
            exercising sharded/`pmap` code paths on a CPU-only machine — it does
            **not** make a single-device run faster (XLA:CPU already parallelises
            each op across cores internally), so it stays opt-in.

    Calling this after JAX has initialised its backend cannot take effect; we warn
    rather than raise so an embedding caller (a notebook, a test) is not broken by it.
    """
    import sys

    if device and device != "auto":
        device = device.lower()
        if device not in _SUPPORTED_DTYPES:
            raise ConfigError(
                f"unknown device {device!r}; expected one of "
                f"{', '.join(sorted(_SUPPORTED_DTYPES))} or 'auto'"
            )
        os.environ["JAX_PLATFORMS"] = device

    if cpu_devices and cpu_devices > 1:
        flag = f"--xla_force_host_platform_device_count={int(cpu_devices)}"
        existing = os.environ.get("XLA_FLAGS", "")
        if "xla_force_host_platform_device_count" not in existing:
            os.environ["XLA_FLAGS"] = f"{existing} {flag}".strip()

    if "jax" in sys.modules:
        logger.warning(
            "configure_backend() ran after jax was already imported — "
            "XLA_FLAGS/JAX_PLATFORMS changes will not take effect for this process"
        )


def describe_backend(compute_dtype: str = "float32") -> BackendInfo:
    """Introspect the live JAX backend. Requires JAX to be importable."""
    import jax

    devices = jax.devices()
    return BackendInfo(
        platform=jax.default_backend(),
        device_count=len(devices),
        device_kind=getattr(devices[0], "device_kind", "unknown") if devices else "none",
        host_cores=os.cpu_count(),
        jax_version=jax.__version__,
        compute_dtype=compute_dtype,
    )


def resolve_dtype(requested: str, platform: str) -> str:
    """Validate a config's ``model.dtype`` against what the platform can run.

    Returns the dtype to actually use. Raises `ConfigError` for a dtype the backend
    has no kernels for (float16 on CPU), because silently upcasting would make the
    run's logged precision a lie — and reproducibility is a roadmap §5 requirement.
    """
    requested = (requested or "float32").lower()
    aliases = {"bf16": "bfloat16", "fp32": "float32", "fp16": "float16", "half": "float16"}
    requested = aliases.get(requested, requested)
    allowed = _SUPPORTED_DTYPES.get(platform, _SUPPORTED_DTYPES["cpu"])
    if requested not in allowed:
        raise ConfigError(
            f"model.dtype={requested!r} is not supported on {platform} "
            f"(supported: {', '.join(allowed)}). "
            f"For {platform} training use bfloat16 (mixed precision) or float32."
        )
    return requested


def estimate_runtime(
    platform: str,
    tokens_per_step: int,
    max_steps: int,
    tokens_per_sec: Optional[float] = None,
) -> float:
    """Estimated wall-clock seconds for a run. Rough by design — see module docstring."""
    rate = tokens_per_sec or _TOKENS_PER_SEC_HINT.get(platform, _TOKENS_PER_SEC_HINT["cpu"])
    return tokens_per_step * max_steps / max(rate, 1e-9)


def format_duration(seconds: float) -> str:
    """``4512`` -> ``"1h 15m"``. Used in the pre-flight banner."""
    seconds = max(0.0, float(seconds))
    if seconds < 90:
        return f"{seconds:.0f}s"
    minutes, sec = divmod(int(seconds), 60)
    if minutes < 90:
        return f"{minutes}m {sec}s"
    hours, minutes = divmod(minutes, 60)
    if hours < 48:
        return f"{hours}h {minutes}m"
    return f"{hours // 24}d {hours % 24}h"


def log_preflight(
    info: BackendInfo,
    tokens_per_step: int,
    max_steps: int,
    params_millions: float,
    warn_over_seconds: float = 6 * 3600,
) -> float:
    """Log the device/precision/ETA banner; warn loudly on absurdly long CPU runs.

    A CPU run that would take days is almost always a config mistake (a GPU config
    pointed at a laptop), and the failure mode is the worst kind: it looks like it is
    working. Returns the estimate in seconds so callers can log it as a metric.
    """
    eta = estimate_runtime(info.platform, tokens_per_step, max_steps)
    logger.info(
        "backend=%s x%d (%s) jax=%s dtype=%s cores=%s",
        info.platform,
        info.device_count,
        info.device_kind,
        info.jax_version,
        info.compute_dtype,
        info.host_cores,
    )
    logger.info(
        "~%.1fM params · %s tok/step · %s steps · rough ETA %s",
        params_millions,
        f"{tokens_per_step:,}",
        f"{max_steps:,}",
        format_duration(eta),
    )
    if info.platform == "cpu" and eta > warn_over_seconds:
        logger.warning(
            "this CPU run is estimated at %s. For a CPU box use the nano tier: "
            "--config src/adhan_slm/configs/adhan_slm_nano_cpu.yaml "
            "(or lower train.max_steps / train.batch_size / model.max_seq_len).",
            format_duration(eta),
        )
    return eta

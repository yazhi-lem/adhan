"""Unit tests for backend/precision resolution (`adhan_slm.training.device`).

Pure-python: no JAX needed except for the one `describe_backend` test, which skips
if the stack is absent. Run standalone with:

    PYTHONPATH=src python -m adhan_slm.training.device_tests
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from adhan_slm.core.exceptions import ConfigError  # noqa: E402
from adhan_slm.core.selftest import run_module_tests, skip_unless  # noqa: E402
from adhan_slm.training.device import (  # noqa: E402
    configure_backend,
    describe_backend,
    estimate_runtime,
    format_duration,
    resolve_dtype,
)


def test_resolve_dtype_accepts_cpu_mixed_precision():
    # bf16 is ~1.5x faster than fp32 on XLA:CPU (measured), so it stays the CPU default.
    assert resolve_dtype("bfloat16", "cpu") == "bfloat16"
    assert resolve_dtype("float32", "cpu") == "float32"


def test_resolve_dtype_normalises_aliases():
    assert resolve_dtype("bf16", "cpu") == "bfloat16"
    assert resolve_dtype("FP32", "gpu") == "float32"
    assert resolve_dtype("half", "gpu") == "float16"


def test_resolve_dtype_rejects_float16_on_cpu():
    # Silently upcasting would make the run's logged precision a lie.
    try:
        resolve_dtype("float16", "cpu")
    except ConfigError as e:
        assert "not supported on cpu" in str(e)
    else:  # pragma: no cover
        raise AssertionError("float16 on CPU should raise ConfigError")


def test_resolve_dtype_allows_float16_on_gpu():
    assert resolve_dtype("float16", "gpu") == "float16"


def test_estimate_runtime_scales_with_work():
    one = estimate_runtime("cpu", tokens_per_step=1024, max_steps=100)
    ten = estimate_runtime("cpu", tokens_per_step=1024, max_steps=1000)
    assert 9.5 < ten / one < 10.5
    # A measured live rate must override the platform hint.
    assert estimate_runtime("cpu", 1024, 100, tokens_per_sec=1024) == 100.0


def test_estimate_runtime_gpu_is_faster_than_cpu():
    assert estimate_runtime("gpu", 1024, 1000) < estimate_runtime("cpu", 1024, 1000)


def test_format_duration_units():
    assert format_duration(45) == "45s"
    assert format_duration(600) == "10m 0s"
    assert format_duration(3 * 3600 + 25 * 60) == "3h 25m"
    assert format_duration(50 * 3600) == "2d 2h"


def test_configure_backend_rejects_unknown_device():
    try:
        configure_backend("cuda")  # the platform is called "gpu" in JAX
    except ConfigError as e:
        assert "unknown device" in str(e)
    else:  # pragma: no cover
        raise AssertionError("unknown device should raise ConfigError")


def test_configure_backend_sets_env():
    saved = {k: os.environ.get(k) for k in ("JAX_PLATFORMS", "XLA_FLAGS")}
    try:
        os.environ.pop("XLA_FLAGS", None)
        configure_backend("cpu", cpu_devices=4)
        assert os.environ["JAX_PLATFORMS"] == "cpu"
        assert "--xla_force_host_platform_device_count=4" in os.environ["XLA_FLAGS"]
        # "auto" must leave accelerator discovery alone rather than pinning CPU.
        os.environ.pop("JAX_PLATFORMS")
        configure_backend("auto")
        assert "JAX_PLATFORMS" not in os.environ
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _has_jax() -> bool:
    try:
        import jax  # noqa: F401
    except ImportError:
        return False
    return True


@skip_unless(_has_jax(), "jax not installed")
def test_describe_backend_reports_a_real_device():
    info = describe_backend("bfloat16")
    assert info.platform in ("cpu", "gpu", "tpu")
    assert info.device_count >= 1
    assert info.compute_dtype == "bfloat16"
    assert "runtime.platform" in info.as_params()


if __name__ == "__main__":
    run_module_tests(globals(), "training backend/precision resolution")

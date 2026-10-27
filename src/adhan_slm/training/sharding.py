"""JAX data-parallel sharding utilities for Adhan SLM.

Provides a minimal, dependency-light abstraction over ``jax.sharding`` that:
  * builds a 1-D data-parallel ``Mesh`` from the available devices,
  * replicates model params across the mesh,
  * shards a batch along the leading (batch) axis,
  * folds a sharded loss back to a scalar via ``jax.lax.pmean``.

The goal is to make the training loop distributed *without* changing its
structure: when a single device is present (or sharding is disabled), every
helper degrades to a no-op so CPU/GPU/TPU single-device runs behave exactly as
before.

Design notes
------------
* We only implement pure **data parallelism** here. Each device holds a full
  replica of the params and processes ``batch / n_devices`` examples. That is
  the correct first step and covers nano/tiny comfortably. Tensor/pipeline
  parallelism (and FSDP for mini) is a later concern; see factory/RUNWAY.md.
* ``jax.sharding`` is imported lazily so importing this module never forces a
  backend to initialise (same contract as the rest of the training package).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from adhan_slm.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ShardingInfo:
    """Describes the mesh a training run is using (or ``None`` for single-device)."""

    enabled: bool
    n_devices: int = 1
    axis_names: tuple = ("dp",)
    device_kind: str = ""

    def as_params(self) -> dict:
        return {
            "device.n_devices": self.n_devices,
            "device.data_parallel": self.enabled,
            "device.axis_names": ",".join(self.axis_names),
        }


def make_data_parallel_mesh(
    n_devices: Optional[int] = None,
    axis_name: str = "dp",
) -> tuple[Any, ShardingInfo]:
    """Build a 1-D data-parallel mesh over the available devices.

    Args:
        n_devices: Number of devices to use. ``None`` or ``0`` means "use all
            locally visible devices". ``1`` disables sharding.
        axis_name: Name of the batch-parallel axis.

    Returns:
        ``(mesh_or_None, info)``. ``mesh`` is ``None`` when sharding is disabled
        (single device or explicit opt-out), which callers treat as the
        single-device path.
    """
    import jax

    all_devices = jax.devices()
    available = len(all_devices)

    if n_devices is None or n_devices <= 0:
        n_devices = available
    n_devices = min(int(n_devices), available)

    if n_devices <= 1:
        return None, ShardingInfo(
            enabled=False,
            n_devices=1,
            axis_names=(axis_name,),
            device_kind=all_devices[0].device_kind if all_devices else "",
        )

    from jax.sharding import Mesh

    devices = all_devices[:n_devices]
    mesh = Mesh(devices, axis_names=(axis_name,))
    info = ShardingInfo(
        enabled=True,
        n_devices=n_devices,
        axis_names=(axis_name,),
        device_kind=devices[0].device_kind,
    )
    logger.info(
        "data-parallel mesh: %d x %s (axis=%s)",
        n_devices,
        info.device_kind,
        axis_name,
    )
    return mesh, info


def shard_batch(batch, mesh, axis_name: str = "dp"):
    """Shard a batch array along its leading axis across ``mesh``.

    Returns the input unchanged when ``mesh is None``.
    """
    if mesh is None:
        return batch
    import jax
    from jax.sharding import NamedSharding
    from jax.sharding import PartitionSpec as P

    sharding = NamedSharding(mesh, P(axis_name))
    return jax.device_put(batch, sharding)


def replicate_params(params, mesh, axis_name: str = "dp"):
    """Replicate params across ``mesh`` (every device holds a full copy)."""
    if mesh is None:
        return params
    import jax
    from jax.sharding import NamedSharding
    from jax.sharding import PartitionSpec as P

    sharding = NamedSharding(mesh, P())
    return jax.device_put(params, sharding)


def all_reduce_mean(value, axis_name: str = "dp", enabled: bool = True):
    """Average ``value`` across the data-parallel axis.

    ``jax.lax.pmean`` is linear, so wrapping the *loss* in it before
    ``jax.value_and_grad`` also all-reduces the gradients for free — this is the
    standard single-program-multiple-data (SPMD) trick and avoids a manual
    ``pmean`` of every gradient leaf.

    Returns ``value`` unchanged when ``enabled`` is False, so the single-device
    path is unaffected.
    """
    if not enabled:
        return value
    import jax

    return jax.lax.pmean(value, axis_name=axis_name)


def global_batch_size(batch_size: int, info: ShardingInfo) -> int:
    """Per-device batch size needed to preserve a global batch across ``n`` devices."""
    if not info.enabled or info.n_devices <= 1:
        return batch_size
    return max(1, batch_size // info.n_devices)

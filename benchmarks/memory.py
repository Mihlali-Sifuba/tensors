"""Host and device memory characterization.

Memory is measured in a separate pass from timing. Tracing allocations
perturbs timing badly enough that mixing the two would corrupt both, so a
case that asks for memory data is run again under instrumentation.
"""

from __future__ import annotations

import gc
import tracemalloc
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass
class MemorySample:
    """Allocation behavior of one measured callable."""

    host_current_bytes: int = 0
    host_peak_bytes: int = 0
    host_allocation_count: int = 0
    device_used_delta_bytes: int | None = None
    device_pool_delta_bytes: int | None = None
    device_peak_used_bytes: int | None = None
    #: Per-call growth that survived the whole repetition batch.
    host_retained_bytes_per_call: float = 0.0
    device_retained_bytes_per_call: float | None = None
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        """Return a serializable view."""
        return {
            "host_current_bytes": self.host_current_bytes,
            "host_peak_bytes": self.host_peak_bytes,
            "host_allocation_count": self.host_allocation_count,
            "host_retained_bytes_per_call": self.host_retained_bytes_per_call,
            "device_used_delta_bytes": self.device_used_delta_bytes,
            "device_pool_delta_bytes": self.device_pool_delta_bytes,
            "device_peak_used_bytes": self.device_peak_used_bytes,
            "device_retained_bytes_per_call": (self.device_retained_bytes_per_call),
            "notes": list(self.notes),
        }


def _device_pool() -> Any | None:
    """Return the active CuPy memory pool, or ``None`` off CUDA."""
    try:
        import cupy
    except ImportError:
        return None
    return cupy.get_default_memory_pool()


def _synchronize() -> None:
    try:
        import cupy
    except ImportError:
        return
    cupy.cuda.get_current_stream().synchronize()


def measure_memory(
    run: Callable[[], Any],
    *,
    backend: str,
    repetitions: int = 8,
) -> MemorySample:
    """Measure one call's allocations and the growth left by ``repetitions``.

    Two facts are wanted and they need different observations. A single
    traced call shows what the operation allocates and how many allocations
    it makes. A batch of calls, compared before and after with collection
    forced, shows what it fails to release.
    """
    sample = MemorySample()
    on_device = backend == "cuda"
    pool = _device_pool() if on_device else None

    # --- one traced call: what does this operation allocate? ---
    gc.collect()
    if pool is not None:
        pool.free_all_blocks()
        _synchronize()
        device_before = pool.used_bytes()
        pool_before = pool.total_bytes()

    tracemalloc.start()
    baseline = tracemalloc.take_snapshot()
    try:
        run()
        if on_device:
            _synchronize()
        current, peak = tracemalloc.get_traced_memory()
        snapshot = tracemalloc.take_snapshot()
    finally:
        tracemalloc.stop()

    differences = snapshot.compare_to(baseline, "lineno")
    sample.host_current_bytes = current
    sample.host_peak_bytes = peak
    sample.host_allocation_count = sum(
        max(entry.count_diff, 0) for entry in differences
    )

    if pool is not None:
        _synchronize()
        sample.device_used_delta_bytes = pool.used_bytes() - device_before
        sample.device_pool_delta_bytes = pool.total_bytes() - pool_before
        sample.device_peak_used_bytes = pool.used_bytes()

    # --- a batch of calls: what does it retain? ---
    gc.collect()
    if pool is not None:
        pool.free_all_blocks()
        _synchronize()
        retained_device_before = pool.used_bytes()
    tracemalloc.start()
    retained_baseline = tracemalloc.take_snapshot()
    try:
        for _ in range(repetitions):
            run()
        if on_device:
            _synchronize()
        gc.collect()
        retained_current, _ = tracemalloc.get_traced_memory()
        retained_snapshot = tracemalloc.take_snapshot()
    finally:
        tracemalloc.stop()
    retained_total = sum(
        entry.size_diff
        for entry in retained_snapshot.compare_to(retained_baseline, "lineno")
    )
    sample.host_retained_bytes_per_call = retained_total / repetitions
    if pool is not None:
        _synchronize()
        sample.device_retained_bytes_per_call = (
            pool.used_bytes() - retained_device_before
        ) / repetitions
    if retained_current < 0:
        sample.notes.append("negative traced retention; collector reclaimed")
    return sample


def device_memory_status() -> dict[str, Any] | None:
    """Return CUDA pool and device occupancy, or ``None`` off CUDA."""
    try:
        import cupy
    except ImportError:
        return None
    pool = cupy.get_default_memory_pool()
    free_bytes, total_bytes = cupy.cuda.Device().mem_info
    return {
        "pool_used_bytes": pool.used_bytes(),
        "pool_total_bytes": pool.total_bytes(),
        "device_free_bytes": free_bytes,
        "device_total_bytes": total_bytes,
    }


def release_device_memory() -> None:
    """Return cached device blocks so a later group starts from a clean pool."""
    try:
        import cupy
    except ImportError:
        return
    cupy.cuda.get_current_stream().synchronize()
    cupy.get_default_memory_pool().free_all_blocks()
    cupy.get_default_pinned_memory_pool().free_all_blocks()


__all__ = [
    "MemorySample",
    "device_memory_status",
    "measure_memory",
    "release_device_memory",
]

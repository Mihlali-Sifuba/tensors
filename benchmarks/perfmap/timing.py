"""Timing primitives with an explicit synchronization policy.

Three questions are answered separately because they have different answers
on an asynchronous backend:

``submit_seconds``
    Host time to return from the measured calls. On CUDA this is submission
    latency *plus* any synchronization the implementation performs itself,
    which is exactly what makes hidden host/device barriers visible.
``total_seconds``
    Host time until the device finished the submitted work. This is the
    latency a caller observes when it needs the value.
``device_seconds``
    GPU execution time measured with CUDA events, independent of host cost.

Synchronization is inserted only at batch boundaries, never between the
measured calls, so an implementation that does not synchronize is never
charged for one.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


#: The synchronization policy every timer in this module implements.
SYNC_POLICY = (
    "device idle before the batch; no synchronization between measured "
    "calls; stream synchronized at the batch boundary; CUDA events recorded "
    "outside the measured calls"
)


@dataclass(frozen=True)
class Timing:
    """One timed batch, already divided by the invocation count."""

    submit_seconds: float
    total_seconds: float
    device_seconds: float | None
    #: One isolated call's host return time, measured with an idle device.
    single_submit_seconds: float | None = None
    #: The same call's host time through completion.
    single_total_seconds: float | None = None


class HostTimer:
    """Time synchronous work on the host."""

    backend = "host"

    def prepare(self) -> None:
        """Bring the execution target to a known idle state."""

    def run_batch(self, run: Callable[[], Any], loops: int) -> Timing:
        """Invoke ``run`` ``loops`` times and return per-invocation timing."""
        counter = time.perf_counter
        start = counter()
        for _ in range(loops):
            run()
        elapsed = (counter() - start) / loops
        return Timing(elapsed, elapsed, None)


class CudaTimer:
    """Time asynchronous CUDA work with host clocks and CUDA events."""

    backend = "cuda"

    def __init__(self) -> None:
        import cupy

        self._cupy = cupy
        self._stream = cupy.cuda.get_current_stream()
        self._start = cupy.cuda.Event()
        self._end = cupy.cuda.Event()

    def prepare(self) -> None:
        """Wait for outstanding device work so the batch starts idle."""
        self._stream.synchronize()

    def run_batch(self, run: Callable[[], Any], loops: int) -> Timing:
        """Return submission, completion, and device time for one batch."""
        counter = time.perf_counter
        self._stream.synchronize()
        self._start.record(self._stream)
        start = counter()
        for _ in range(loops):
            run()
        submitted = counter()
        self._end.record(self._stream)
        self._stream.synchronize()
        completed = counter()
        device_milliseconds = self._cupy.cuda.get_elapsed_time(
            self._start,
            self._end,
        )
        single_submit, single_total = self._probe_single(run)
        return Timing(
            (submitted - start) / loops,
            (completed - start) / loops,
            device_milliseconds / 1000.0 / loops,
            single_submit,
            single_total,
        )

    def _probe_single(self, run: Callable[[], Any]) -> tuple[float, float]:
        """Time one call against an idle device, before and after completion.

        With nothing queued, an asynchronous call returns in launch time.
        This is descriptive only; :meth:`probe_synchronization` is what
        decides whether the call blocks.
        """
        counter = time.perf_counter
        self._stream.synchronize()
        start = counter()
        run()
        submitted = counter()
        self._stream.synchronize()
        completed = counter()
        return submitted - start, completed - start

    # -- synchronization detection --------------------------------------

    def _barrier_operands(self) -> tuple[Any, Any]:
        """Return buffers used only to occupy the device."""
        operands = getattr(self, "_barrier", None)
        if operands is None:
            size = 4_000_000
            operands = (
                self._cupy.full(size, 1.5, dtype=self._cupy.float64),
                self._cupy.full(size, 2.0, dtype=self._cupy.float64),
            )
            self._barrier = operands
        return operands

    def probe_synchronization(
        self,
        run: Callable[[], Any],
        *,
        launches: int = 8,
    ) -> dict[str, float | bool | None]:
        """Decide whether ``run`` waits for already-queued device work.

        Comparing submission against completion cannot answer this, because
        a call that merely launches many kernels is host-bound for the same
        reason a blocking one is. Occupying the device first separates them:
        queued work is asynchronous, so a non-blocking call returns while it
        is still running, and only a call that synchronizes has to absorb
        it.
        """
        left, right = self._barrier_operands()
        counter = time.perf_counter

        # A first call can compile a kernel or import a module, which would
        # swamp the comparison. Warm the callable before measuring it.
        for _ in range(3):
            run()
        self._stream.synchronize()

        # How long does the barrier itself keep the device busy?
        self._start.record(self._stream)
        for _ in range(launches):
            self._cupy.add(left, right)
        self._end.record(self._stream)
        self._stream.synchronize()
        barrier_seconds = self._cupy.cuda.get_elapsed_time(
            self._start, self._end
        ) / 1000.0

        # Both sides are taken as a minimum over repeats: host scheduling
        # jitter can only add time, so the smallest observation is the
        # closest to the cost being attributed.
        idle_host = float("inf")
        behind_host = float("inf")
        for _ in range(3):
            self._stream.synchronize()
            start = counter()
            run()
            idle_host = min(idle_host, counter() - start)
            self._stream.synchronize()

            for _ in range(launches):
                self._cupy.add(left, right)
            start = counter()
            run()
            behind_host = min(behind_host, counter() - start)
            self._stream.synchronize()

        absorbed = behind_host - idle_host
        return {
            "barrier_device_seconds": barrier_seconds,
            "call_host_idle_seconds": idle_host,
            "call_host_behind_barrier_seconds": behind_host,
            "absorbed_seconds": absorbed,
            "absorbed_fraction_of_barrier": (
                None if barrier_seconds <= 0.0 else absorbed / barrier_seconds
            ),
            # Absorbing most of the queued work is only possible by waiting
            # for it. Half the barrier is a wide margin against host jitter.
            "synchronizes": (
                barrier_seconds > 0.0
                and absorbed > 0.5 * barrier_seconds
            ),
        }


def timer_for(backend: str) -> HostTimer | CudaTimer:
    """Return the timer that matches a backend's execution model."""
    if backend == "cuda":
        return CudaTimer()
    return HostTimer()


__all__ = ["SYNC_POLICY", "CudaTimer", "HostTimer", "Timing", "timer_for"]

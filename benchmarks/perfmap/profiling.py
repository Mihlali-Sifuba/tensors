"""Diagnostic profiling. Not measurement.

These tools answer *where* a cost identified by the benchmarks comes from.
They are deliberately intrusive — they wrap the provider module the kernels
call and count the conversions CuPy performs — so nothing here produces a
timing anyone should compare against a benchmark result.

The instrumentation is installed and removed around a single call. It patches
benchmark-side references only; no file under ``tensors/`` is modified.

What each probe answers:

``provider_calls``
    which provider primitives one public operation invokes, and how many
    times. This is how "one operation, many kernels" is established, and
    how copies and materializations are located by name.
``host_transfers``
    how many times a device value is converted to a host value during one
    call. This is the count behind a synchronization the barrier probe has
    already proven.
``hot_functions``
    which Python functions consume the fixed overhead, by cProfile.
``line_costs``
    the same, attributed to the internal call chain rather than to leaves.
"""

from __future__ import annotations

import cProfile
import importlib
import io
import pstats
import types
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any


#: Every kernel module that binds ``_numpy`` as its own global. Each has to
#: be patched separately, because the import is by symbol.
_KERNEL_MODULES: tuple[str, ...] = (
    "tensors.backend.kernels.core",
    "tensors.backend.kernels.creation",
    "tensors.backend.kernels.manipulation",
    "tensors.backend.kernels.conv.backward",
    "tensors.backend.kernels.conv.forward",
    "tensors.backend.kernels.elementwise.binary_ops",
    "tensors.backend.kernels.elementwise.clipping",
    "tensors.backend.kernels.elementwise.comparison_ops",
    "tensors.backend.kernels.elementwise.extrema",
    "tensors.backend.kernels.elementwise.selection",
    "tensors.backend.kernels.elementwise.unary_ops",
    "tensors.backend.kernels.fusion.backward",
    "tensors.backend.kernels.fusion.forward",
    "tensors.backend.kernels.linalg.matmul_ops",
    "tensors.backend.kernels.linalg.outer_ops",
    "tensors.backend.kernels.nn.losses",
    "tensors.backend.kernels.nn.normalization_ops",
    "tensors.backend.kernels.nn.validation",
    "tensors.backend.kernels.optim.adam",
    "tensors.backend.kernels.optim.rmsprop",
    "tensors.backend.kernels.optim.sgd",
    "tensors.backend.kernels.reductions.extrema",
    "tensors.backend.kernels.reductions.logsumexp_ops",
    "tensors.backend.kernels.reductions.reduction_ops",
    "tensors.backend.kernels.reductions.shape",
)

#: Provider primitives that move or duplicate values rather than compute.
COPY_PRIMITIVES = frozenset({
    "asarray", "ascontiguousarray", "array", "copy", "asnumpy",
    "astype", "reshape", "broadcast_arrays", "broadcast_to", "squeeze",
    "expand_dims", "concatenate", "stack", "put_along_axis",
})

#: Provider primitives whose result a guard converts to a Python bool.
GUARD_PRIMITIVES = frozenset({
    "isfinite", "isnan", "all", "any", "signbit", "floor",
})


class _RecordingProvider(types.ModuleType):
    """A provider module that counts the primitives called through it."""

    def __init__(self, real: Any, counts: dict[str, int]) -> None:
        super().__init__(real.__name__)
        object.__setattr__(self, "_real", real)
        object.__setattr__(self, "_counts", counts)
        object.__setattr__(self, "_wrapped", {})

    def __getattr__(self, name: str) -> Any:
        cached = self._wrapped.get(name)
        if cached is not None:
            return cached
        value = getattr(self._real, name)
        # Types, dtypes and submodules are passed through untouched: only
        # callables that perform work are counted.
        if isinstance(value, type) or not callable(value):
            if isinstance(value, types.ModuleType):
                value = _RecordingProvider(value, self._counts)
                prefix = name
                # A submodule's calls are reported under "module.function".
                object.__setattr__(value, "_prefix", prefix)
            self._wrapped[name] = value
            return value

        prefix = getattr(self, "_prefix", "")
        label = f"{prefix}.{name}" if prefix else name
        counts = self._counts

        def wrapper(*arguments: Any, **keywords: Any) -> Any:
            counts[label] = counts.get(label, 0) + 1
            return value(*arguments, **keywords)

        wrapper.__name__ = name
        self._wrapped[name] = wrapper
        return wrapper


@contextmanager
def _recording_provider(backend: str) -> Iterator[dict[str, int]]:
    """Install a counting provider in every kernel module."""
    real = importlib.import_module("cupy" if backend == "cuda" else "numpy")
    counts: dict[str, int] = {}
    proxy = _RecordingProvider(real, counts)

    def provider() -> Any:
        return proxy

    patched: list[tuple[Any, Any]] = []
    for name in _KERNEL_MODULES:
        try:
            module = importlib.import_module(name)
        except ImportError:
            continue
        if hasattr(module, "_numpy"):
            patched.append((module, module._numpy))
            module._numpy = provider
    try:
        yield counts
    finally:
        for module, original in patched:
            module._numpy = original


def provider_calls(
    run: Callable[[], Any],
    *,
    backend: str,
    warm: bool = True,
) -> dict[str, Any]:
    """Return which provider primitives one call invokes, and how often.

    A first call can compile a kernel or resolve a cache, so the callable is
    warmed before the counted invocation unless asked otherwise.
    """
    if warm:
        run()
    with _recording_provider(backend) as counts:
        run()
        recorded = dict(counts)
    total = sum(recorded.values())
    return {
        "total_calls": total,
        "calls": dict(
            sorted(recorded.items(), key=lambda item: (-item[1], item[0]))
        ),
        "copy_calls": {
            name: count for name, count in recorded.items()
            if name.rsplit(".", 1)[-1] in COPY_PRIMITIVES
        },
        "guard_calls": {
            name: count for name, count in recorded.items()
            if name.rsplit(".", 1)[-1] in GUARD_PRIMITIVES
        },
    }


@contextmanager
def _counting_host_transfers() -> Iterator[dict[str, int]]:
    """Count the CuPy conversions that move a device value to the host."""
    import cupy

    counts: dict[str, int] = {}
    array_type = cupy.ndarray
    original: dict[str, Any] = {}

    def install(name: str) -> None:
        attribute = getattr(array_type, name, None)
        if attribute is None:
            return
        original[name] = attribute

        def wrapper(self: Any, *arguments: Any, **keywords: Any) -> Any:
            counts[name] = counts.get(name, 0) + 1
            return attribute(self, *arguments, **keywords)

        try:
            setattr(array_type, name, wrapper)
        except (AttributeError, TypeError):
            original.pop(name, None)

    for name in ("__bool__", "__float__", "__int__", "item", "get",
                 "tolist", "__index__"):
        install(name)

    module_original = cupy.asnumpy

    def asnumpy(*arguments: Any, **keywords: Any) -> Any:
        counts["asnumpy"] = counts.get("asnumpy", 0) + 1
        return module_original(*arguments, **keywords)

    cupy.asnumpy = asnumpy
    try:
        yield counts
    finally:
        for name, attribute in original.items():
            setattr(array_type, name, attribute)
        cupy.asnumpy = module_original


def host_transfers(
    run: Callable[[], Any],
    *,
    warm: bool = True,
) -> dict[str, Any]:
    """Return how many device-to-host conversions one call performs.

    Each of these is a synchronization: the host cannot produce a Python
    value from a device buffer without waiting for the work that fills it.
    """
    if warm:
        run()
    with _counting_host_transfers() as counts:
        run()
        recorded = dict(counts)
    return {
        "total_transfers": sum(recorded.values()),
        "transfers": dict(
            sorted(recorded.items(), key=lambda item: (-item[1], item[0]))
        ),
    }


def hot_functions(
    run: Callable[[], Any],
    *,
    repeats: int = 200,
    limit: int = 25,
    inside: str | None = "tensors",
) -> dict[str, Any]:
    """Return the functions that consume one path's time, by cProfile.

    ``inside`` restricts the report to functions whose file path contains
    that fragment, which is what makes a fixed-overhead question answerable:
    the provider call is one leaf, and everything else is the library.
    """
    profiler = cProfile.Profile()
    run()  # warm
    profiler.enable()
    for _ in range(repeats):
        run()
    profiler.disable()

    statistics = pstats.Stats(profiler)
    rows: list[dict[str, Any]] = []
    for (path, line, function), values in statistics.stats.items():
        call_count, _, total_time, cumulative_time, _ = values
        rows.append({
            "function": function,
            "location": f"{path}:{line}",
            "calls": call_count,
            "calls_per_invocation": call_count / repeats,
            "total_seconds": total_time,
            "cumulative_seconds": cumulative_time,
            "self_seconds_per_invocation": total_time / repeats,
        })
    library = [
        row for row in rows
        if inside is None or inside in row["location"]
    ]
    library.sort(key=lambda row: -row["total_seconds"])
    overall = sorted(rows, key=lambda row: -row["total_seconds"])

    buffer = io.StringIO()
    pstats.Stats(profiler, stream=buffer).sort_stats("tottime").print_stats(
        limit
    )
    return {
        "repeats": repeats,
        "library_functions": library[:limit],
        "all_functions": overall[:limit],
        "total_self_seconds": sum(row["total_seconds"] for row in rows),
        "text_report": buffer.getvalue(),
    }


def summarize_provider_calls(report: dict[str, Any]) -> str:
    """Return a one-line description of a provider-call report."""
    calls = report["calls"]
    top = ", ".join(
        f"{name}x{count}" for name, count in list(calls.items())[:6]
    )
    return f"{report['total_calls']} provider calls: {top}"


__all__ = [
    "COPY_PRIMITIVES",
    "GUARD_PRIMITIVES",
    "host_transfers",
    "hot_functions",
    "provider_calls",
    "summarize_provider_calls",
]

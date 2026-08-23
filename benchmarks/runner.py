"""Reusable timing and reporting utilities for the benchmark suite."""

from __future__ import annotations

import gc
import json
import platform
import statistics
import subprocess
import sys
import timeit
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tensors import get_backend


@dataclass(frozen=True)
class BenchmarkCase:
    """One independently calibrated public-API benchmark."""

    name: str
    run: Callable[[], object]
    validate: Callable[[], None] | None = None
    work_items: int | None = None
    gc_enabled: bool = False
    description: str = ""


def _calibrate(timer: timeit.Timer, target_seconds: float) -> int:
    """Choose a loop count whose sample approaches the target duration."""
    loops = 1
    maximum_loops = 10_000_000
    while True:
        elapsed = timer.timeit(number=loops)
        if elapsed >= target_seconds or loops >= maximum_loops:
            return loops
        if elapsed <= 0.0:
            scale = 10
        else:
            scale = max(2, min(10, int(target_seconds / elapsed)))
        loops = min(loops * scale, maximum_loops)


def _timer(case: BenchmarkCase) -> timeit.Timer:
    """Create a Timer with consistent garbage-collection behavior."""
    setup = gc.enable if case.gc_enabled else (lambda: None)

    def run_and_synchronize() -> object:
        result = case.run()
        _synchronize_backend()
        return result

    return timeit.Timer(run_and_synchronize, setup=setup)


def _synchronize_backend() -> None:
    """Wait for asynchronous backend work before recording elapsed time."""
    if get_backend() != "cuda":
        return
    import cupy

    cupy.cuda.get_current_stream().synchronize()


def measure(
    case: BenchmarkCase,
    *,
    repeats: int,
    target_seconds: float,
) -> dict[str, Any]:
    """Validate, calibrate, and measure one benchmark case."""
    if case.validate is not None:
        case.validate()
    _synchronize_backend()

    gc.collect()
    timer = _timer(case)
    loops = _calibrate(timer, target_seconds)

    samples = []
    for _ in range(repeats):
        gc.collect()
        samples.append(timer.timeit(number=loops) / loops)

    median_seconds = statistics.median(samples)
    deviations = [abs(sample - median_seconds) for sample in samples]
    median_absolute_deviation = statistics.median(deviations)
    variability_percent = (
        0.0
        if median_seconds == 0.0
        else median_absolute_deviation / median_seconds * 100.0
    )

    result: dict[str, Any] = {
        "description": case.description,
        "loops_per_sample": loops,
        "median_seconds": median_seconds,
        "minimum_seconds": min(samples),
        "median_absolute_deviation_seconds": median_absolute_deviation,
        "variability_percent": variability_percent,
        "samples_seconds": samples,
    }
    if case.work_items is not None:
        result["work_items"] = case.work_items
        result["work_items_per_second"] = (
            None if median_seconds == 0.0 else case.work_items / median_seconds
        )
    return result


def _git_commit() -> str | None:
    """Return the current short commit when running inside the repository."""
    repository = Path(__file__).resolve().parents[1]
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip() or None


def _git_dirty() -> bool | None:
    """Report whether tracked or untracked repository files are modified."""
    repository = Path(__file__).resolve().parents[1]
    try:
        completed = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return bool(completed.stdout.strip())


def environment_metadata() -> dict[str, Any]:
    """Return enough context to make a benchmark result interpretable."""
    metadata = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "commit": _git_commit(),
        "git_dirty": _git_dirty(),
        "backend": get_backend(),
        "python": sys.version,
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
    }
    if get_backend() == "cuda":
        import cupy

        device = cupy.cuda.Device()
        properties = cupy.cuda.runtime.getDeviceProperties(device.id)
        name = properties["name"]
        if isinstance(name, bytes):
            name = name.decode("utf-8")
        metadata["cupy"] = cupy.__version__
        metadata["cuda_runtime"] = cupy.cuda.runtime.runtimeGetVersion()
        metadata["cuda_device"] = name
        metadata["cuda_device_id"] = device.id
    return metadata


def run_suite(
    cases: Iterable[BenchmarkCase],
    *,
    repeats: int,
    target_seconds: float,
) -> dict[str, Any]:
    """Run cases in order and return a serializable report."""
    results: dict[str, Any] = {}
    for case in cases:
        print(f"Running [{get_backend()}] {case.name}...", flush=True)
        results[case.name] = measure(
            case,
            repeats=repeats,
            target_seconds=target_seconds,
        )
    return {
        "metadata": environment_metadata(),
        "settings": {
            "repeats": repeats,
            "target_seconds_per_sample": target_seconds,
        },
        "benchmarks": results,
    }


def combine_backend_reports(
    reports: Mapping[str, dict[str, Any]],
) -> dict[str, Any]:
    """Combine independently measured backend reports for serialization."""
    if not reports:
        raise ValueError("at least one backend report is required")
    first_report = next(iter(reports.values()))
    return {
        "metadata": {
            "backends": list(reports),
        },
        "settings": first_report["settings"],
        "backends": dict(reports),
    }


def _duration(seconds: float) -> str:
    if seconds < 1e-6:
        return f"{seconds * 1e9:.2f} ns"
    if seconds < 1e-3:
        return f"{seconds * 1e6:.2f} us"
    if seconds < 1.0:
        return f"{seconds * 1e3:.2f} ms"
    return f"{seconds:.2f} s"


def _throughput(value: float | None) -> str:
    if value is None:
        return "-"
    if value >= 1e9:
        return f"{value / 1e9:.2f} G/s"
    if value >= 1e6:
        return f"{value / 1e6:.2f} M/s"
    if value >= 1e3:
        return f"{value / 1e3:.2f} K/s"
    return f"{value:.2f}/s"


def print_report(report: dict[str, Any]) -> None:
    """Print a compact comparison-friendly result table."""
    benchmarks = report["benchmarks"]
    name_width = max([len("benchmark"), *(len(name) for name in benchmarks)])
    heading = (
        f"{'benchmark':<{name_width}}  {'median':>10}  "
        f"{'MAD':>8}  {'work':>12}"
    )
    print(f"\nBackend: {report['metadata']['backend']}")
    print(heading)
    print("-" * len(heading))
    for name, result in benchmarks.items():
        print(
            f"{name:<{name_width}}  "
            f"{_duration(result['median_seconds']):>10}  "
            f"{result['variability_percent']:>7.2f}%  "
            f"{_throughput(result.get('work_items_per_second')):>12}"
        )


def print_backend_comparison(report: dict[str, Any]) -> None:
    """Print backend medians, variability, and optional-backend speedups."""
    reports = report["backends"]
    backend_names = list(reports)
    first_benchmarks = reports[backend_names[0]]["benchmarks"]
    name_width = max(
        [len("benchmark"), *(len(name) for name in first_benchmarks)]
    )
    columns = "".join(
        f"  {backend + ' median':>14}  {backend + ' MAD':>10}"
        for backend in backend_names
    )
    speedup_backends = (
        [name for name in backend_names if name != "python"]
        if "python" in reports
        else []
    )
    speedup_headings = "".join(
        f"  {backend.title() + ' speedup':>13}"
        for backend in speedup_backends
    )
    heading = f"{'benchmark':<{name_width}}{columns}{speedup_headings}"

    print("\nBackend comparison")
    print(heading)
    print("-" * len(heading))
    for name in first_benchmarks:
        row = f"{name:<{name_width}}"
        for backend in backend_names:
            result = reports[backend]["benchmarks"][name]
            row += (
                f"  {_duration(result['median_seconds']):>14}"
                f"  {result['variability_percent']:>9.2f}%"
            )
        for backend in speedup_backends:
            python_seconds = reports["python"]["benchmarks"][name][
                "median_seconds"
            ]
            backend_seconds = reports[backend]["benchmarks"][name][
                "median_seconds"
            ]
            speedup = (
                float("inf")
                if backend_seconds == 0.0
                else python_seconds / backend_seconds
            )
            row += f"  {speedup:>12.2f}x"
        print(row)


def write_report(report: dict[str, Any], output: Path) -> None:
    """Write a JSON result file selected explicitly by the caller."""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

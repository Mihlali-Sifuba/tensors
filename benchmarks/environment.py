"""System, toolchain, and repository metadata that makes a run interpretable."""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_REPOSITORY = Path(__file__).resolve().parents[2]


def _git(*arguments: str) -> str | None:
    """Return trimmed ``git`` output, or ``None`` when it cannot be read."""
    try:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=_REPOSITORY,
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip() or None


def git_metadata() -> dict[str, Any]:
    """Return the exact repository state the measurements describe."""
    status = _git("status", "--porcelain")
    return {
        "commit": _git("rev-parse", "HEAD"),
        "commit_short": _git("rev-parse", "--short", "HEAD"),
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "describe": _git("describe", "--always", "--dirty"),
        "dirty": None if status is None else bool(status),
        "dirty_paths": (
            [] if not status else sorted(line[3:] for line in status.splitlines())
        ),
        "upstream_commit": _git("rev-parse", "origin/main"),
    }


def _cpu_metadata() -> dict[str, Any]:
    """Return processor identification available without extra dependencies."""
    return {
        "processor": platform.processor(),
        "machine": platform.machine(),
        "logical_cores": os.cpu_count(),
        "architecture": platform.architecture()[0],
    }


def _numpy_metadata() -> dict[str, Any] | None:
    try:
        import numpy
    except ImportError:
        return None
    return {
        "version": numpy.__version__,
        "blas": _blas_name(numpy),
    }


def _blas_name(numpy: Any) -> str | None:
    """Return the BLAS implementation NumPy was built against, if reported."""
    config = getattr(numpy, "show_config", None)
    if config is None:
        return None
    try:
        information = config(mode="dicts")
    except (TypeError, ValueError):
        return None
    build = information.get("Build Dependencies", {})
    blas = build.get("blas", {})
    return blas.get("name")


def _cuda_metadata() -> dict[str, Any] | None:
    try:
        import cupy
    except ImportError:
        return None
    try:
        device = cupy.cuda.Device()
        properties = cupy.cuda.runtime.getDeviceProperties(device.id)
    except Exception as error:  # noqa: BLE001 - reported, not handled
        return {"cupy": cupy.__version__, "unavailable": repr(error)}
    name = properties["name"]
    if isinstance(name, bytes):
        name = name.decode("utf-8", "replace")
    free_bytes, total_bytes = device.mem_info
    return {
        "cupy": cupy.__version__,
        "runtime_version": cupy.cuda.runtime.runtimeGetVersion(),
        "driver_version": cupy.cuda.runtime.driverGetVersion(),
        "device_id": device.id,
        "device_name": name,
        "compute_capability": f"{properties['major']}.{properties['minor']}",
        "multiprocessors": properties["multiProcessorCount"],
        "total_memory_bytes": total_bytes,
        "free_memory_bytes": free_bytes,
        "clock_rate_khz": properties.get("clockRate"),
        "memory_clock_khz": properties.get("memoryClockRate"),
        "memory_bus_width": properties.get("memoryBusWidth"),
    }


def environment_metadata() -> dict[str, Any]:
    """Return everything needed to compare this run against another machine."""
    import tensors as ts

    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git": git_metadata(),
        "os": {
            "platform": platform.platform(),
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
        },
        "python": {
            "version": sys.version,
            "version_info": list(sys.version_info[:3]),
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
        },
        "cpu": _cpu_metadata(),
        "numpy": _numpy_metadata(),
        "cuda": _cuda_metadata(),
        "tensors": {
            "available_backends": list(ts.available_backends()),
            "default_backend": ts.get_backend(),
            "environment_backend": os.environ.get("TENSORS_BACKEND"),
        },
    }


__all__ = ["environment_metadata", "git_metadata"]

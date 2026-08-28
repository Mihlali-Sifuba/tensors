"""Fresh-process optional-backend startup benchmarks."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import tensors as ts

from .runner import BenchmarkBackend, BenchmarkCase


_ACCELERATED = frozenset[BenchmarkBackend]({"numpy", "cuda"})


def cases() -> list[BenchmarkCase]:
    """Measure true cold imports and first operations in fresh interpreters."""
    backend = ts.get_backend()
    if backend not in _ACCELERATED:
        return []
    selected = frozenset[BenchmarkBackend]({backend})
    repository = Path(__file__).resolve().parents[1]
    environment = dict(os.environ)
    environment["TENSORS_BACKEND"] = "python"
    provider = "cupy" if backend == "cuda" else "numpy"

    def run_script(script: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, "-c", script],
            cwd=repository,
            env=environment,
            check=True,
            capture_output=True,
            text=True,
        )

    def import_provider():
        return run_script(f"import {provider}; print({provider}.__version__)")

    operation_script = (
        "import tensors as ts; "
        f"ts.set_backend('{backend}'); "
        "x = ts.full((100000,), 1.0); "
        "y = x + 2.0; "
        "print(y[-1])"
    )

    def first_operation():
        return run_script(operation_script)

    def validate(run) -> None:
        completed = run()
        assert completed.returncode == 0
        assert completed.stdout.strip()

    return [
        BenchmarkCase(
            name="startup.provider_import/fresh-process",
            run=import_provider,
            validate=lambda: validate(import_provider),
            work_items=1,
            description="fresh interpreter and optional provider import",
            layer="startup",
            backends=selected,
        ),
        BenchmarkCase(
            name="startup.first_tensor_operation/fresh-process",
            run=first_operation,
            validate=lambda: validate(first_operation),
            work_items=100_000,
            description=(
                "fresh interpreter, backend initialization, and first operation"
            ),
            layer="startup",
            backends=selected,
        ),
    ]

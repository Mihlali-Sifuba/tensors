"""Direct access to the libraries this package is measured against.

A baseline is the other implementation, called on its own terms: NumPy's
``add`` on a ``numpy.ndarray``, not ``tensors`` configured to use NumPy. The
distinction matters because selecting the NumPy backend still pays for a
Tensor, a dtype decision, a dispatch, and a storage wrapper, and the whole
point of a baseline is to show what those cost by leaving them out.

Keeping the adapters here rather than inside the workloads means there is
one definition of "raw NumPy" and one of "raw CuPy", so two suites comparing
against the same library are comparing against the same thing.

Nothing here knows about ``tensors``. A baseline builds its own arrays, in
its own memory, and the workload decides what to do with them.
"""

from __future__ import annotations

from typing import Any

from ..case import Unsupported

#: The backend each baseline is the external counterpart of.
BASELINE_MODULES: dict[str, str] = {
    "numpy": "numpy",
    "cuda": "cupy",
}


def for_backend(backend: str) -> Any:
    """Return the baseline adapter that corresponds to ``backend``.

    The Python backend has no counterpart: its reference kernels *are* the
    implementation, so there is nothing external to compare them against.
    """
    if backend == "numpy":
        from . import numpy as adapter
    elif backend == "cuda":
        from . import cupy as adapter
    else:
        raise Unsupported(
            "the Python backend has no array provider to compare against; "
            "its reference kernels are the implementation"
        )
    return adapter


def module(backend: str) -> Any:
    """Return the external library backing ``backend``."""
    return for_backend(backend).module()


def array(
    backend: str,
    shape: tuple[int, ...],
    *,
    dtype_name: str = "float64",
    kind: str = "ramp",
    value: float = 1.5,
) -> Any:
    """Build a native array holding the same values a Tensor would."""
    return for_backend(backend).array(
        shape, dtype_name=dtype_name, kind=kind, value=value
    )


__all__ = ["BASELINE_MODULES", "array", "for_backend", "module"]

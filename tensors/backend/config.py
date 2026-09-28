"""Process-wide and context-local backend selection.

Availability detection and the ``TENSORS_BACKEND`` default live here because
they exist to answer the selection questions this module exposes.

A function here is named for the operation it performs, not for how widely it
is published: what belongs to the supported API is decided by the package
facades, ``tensors.backend`` and ``tensors``, which re-export the names they
intend to support and leave the rest reachable only through this module.
Selection *state* stays underscore-private, because it is not an operation any
caller may perform: it is mutated only through :func:`set_backend` and
:func:`use_backend`, which hold the lock and clear the kernel cache.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import cast

# Imported as a module, not by symbol: selection changes tell the kernel loader
# to forget its cache, and the loader reads the active selection back from here.
# Binding both sides through the module object keeps that two-way relationship
# working whichever of the two is imported first.
from . import loading
from .types import BackendName, BackendSelection


class BackendUnavailableError(RuntimeError):
    """Raised when an explicitly selected optional backend is unavailable."""


class BackendOperationUnsupportedError(RuntimeError):
    """Raised when an explicitly selected backend cannot execute an operation.

    Explicit selection is an execution requirement (`docs/backends.md`,
    *Execution requirements*), so a backend that cannot produce the required
    result says so rather than letting another backend answer in its place.
    """


class BackendMismatchError(RuntimeError):
    """Raised when an operation receives storage from another backend."""


_VALID_BACKENDS = {"python", "numpy", "cuda", "auto"}
_backend_lock = threading.RLock()
_backend_override: ContextVar[BackendName | None] = ContextVar(
    "tensors_backend_override",
    default=None,
)


def numpy_available() -> bool:
    """Return whether NumPy can be imported in this environment."""
    try:
        return importlib.util.find_spec("numpy") is not None
    except (ImportError, ValueError):
        return False


def cuda_available() -> bool:
    """Return whether CuPy can access at least one CUDA device."""
    try:
        if importlib.util.find_spec("cupy") is None:
            return False
        cupy = importlib.import_module("cupy")
        return bool(cupy.cuda.runtime.getDeviceCount())
    except Exception:
        # Import, driver, and runtime errors all mean the backend cannot execute.
        return False


def available_backends() -> tuple[BackendName, ...]:
    """Return the numerical backends available in this environment."""
    available: list[BackendName] = ["python"]
    if numpy_available():
        available.append("numpy")
    if cuda_available():
        available.append("cuda")
    return tuple(available)


def resolve_backend(backend: str) -> BackendName:
    """Return the concrete backend a selection names, or raise.

    ``"auto"`` resolves here rather than at the point of use, so every
    caller downstream of a selection sees one concrete backend.
    """
    normalized = backend.strip().lower()
    if normalized not in _VALID_BACKENDS:
        choices = ", ".join(sorted(_VALID_BACKENDS))
        raise ValueError(f"Unknown backend {backend!r}; expected one of: {choices}")
    if normalized == "auto":
        return "numpy" if numpy_available() else "python"
    if normalized == "numpy" and not numpy_available():
        raise BackendUnavailableError(
            "The NumPy backend is unavailable. Install it with "
            '`pip install "ms-tensors[numpy]"`.'
        )
    if normalized == "cuda" and not cuda_available():
        raise BackendUnavailableError(
            "The CUDA backend is unavailable. Install the CuPy build matching "
            'your driver with `pip install "ms-tensors[cuda12]"` or '
            '`pip install "ms-tensors[cuda13]"`.'
        )
    return cast(BackendName, normalized)


def environment_default() -> BackendName:
    """Return the backend ``TENSORS_BACKEND`` selects, defaulting to Python."""
    configured = os.environ.get("TENSORS_BACKEND", "python")
    return resolve_backend(configured)


_process_backend = environment_default()


def get_backend() -> BackendName:
    """Return the backend active in the current execution context."""
    override = _backend_override.get()
    if override is not None:
        return override
    with _backend_lock:
        return _process_backend


def set_backend(backend: BackendSelection) -> None:
    """Set the process-wide default backend.

    Configure the default before starting worker threads. Existing scoped
    overrides created by :func:`use_backend` remain active until their contexts
    exit.
    """
    selected = resolve_backend(backend)
    global _process_backend
    with _backend_lock:
        _process_backend = selected
        loading._clear_backend_kernel_cache()


@contextmanager
def use_backend(backend: BackendSelection) -> Iterator[None]:
    """Temporarily select a backend in the current execution context."""
    selected = resolve_backend(backend)
    # Besides keeping the cache bounded across backend changes, clearing here
    # ensures deliberate runtime replacement of an internal kernel (for
    # instrumentation or tests) is observed on entry to the scoped backend.
    loading._clear_backend_kernel_cache()
    token = _backend_override.set(selected)
    try:
        yield
    finally:
        _backend_override.reset(token)
        loading._clear_backend_kernel_cache()


#: The operations this module offers. The facades decide which of them the
#: package supports publicly; the rest stay reachable through this module.
__all__ = [
    "BackendMismatchError",
    "BackendOperationUnsupportedError",
    "BackendUnavailableError",
    "available_backends",
    "cuda_available",
    "environment_default",
    "get_backend",
    "numpy_available",
    "resolve_backend",
    "set_backend",
    "use_backend",
]

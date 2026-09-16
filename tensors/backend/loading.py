"""Lazy loading of a backend's kernel package.

A backend's ``kernels`` package is imported the first time that backend is
used, so selecting Python never imports NumPy or CuPy. Kernel lookups are
cached per backend and name; changing the selection clears that cache, which
is also what makes a kernel replaced at runtime, for instrumentation or a
test, visible to the next call.
"""

from __future__ import annotations

import importlib
from functools import lru_cache
from types import ModuleType
from typing import Any

from tensors.backend import config
from tensors.backend.types import BackendName


@lru_cache(maxsize=3)
def load_backend(backend: BackendName) -> ModuleType:
    """Return the kernel package for ``backend``, importing it on first use."""
    return importlib.import_module(f"tensors.backend.{backend}.kernels")


@lru_cache(maxsize=None)
def _load_backend_kernel(backend: BackendName, name: str) -> Any:
    return getattr(load_backend(backend), name)


def _backend_kernel(name: str) -> Any:
    """Return the named kernel from the backend selected right now."""
    return _load_backend_kernel(config.get_backend(), name)


def _clear_backend_kernel_cache() -> None:
    _load_backend_kernel.cache_clear()

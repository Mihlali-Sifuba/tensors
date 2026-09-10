"""Provider-module and kernel loading.

Given a selected backend and a kernel name, resolve the callable that
implements it. Provider modules are imported on first use, so an optional
dependency stays unimported until a kernel from that backend is requested.
"""

from __future__ import annotations

import importlib
from functools import lru_cache
from typing import Any

from . import config  # imported as a module; see the note in config.py
from .types import BackendName


#: Package holding the provider modules loaded by name below. Kernels live
#: beside this module rather than under it.
_PROVIDER_PACKAGE = __name__.rpartition(".")[0]


@lru_cache(maxsize=None)
def _load_backend_kernel(backend: BackendName, name: str) -> Any:
    """Load and cache one optional-backend kernel callable."""
    if backend not in {"numpy", "cuda"}:
        raise RuntimeError("A kernel was requested for the Python backend")
    module = importlib.import_module(f"{_PROVIDER_PACKAGE}.{backend}")
    return getattr(module, name)


def _backend_kernel(name: str) -> Any:
    """Return a cached kernel from the active optional backend."""
    return _load_backend_kernel(config.get_backend(), name)


def _clear_backend_kernel_cache() -> None:
    """Forget cached callables after an explicit backend-context change."""
    _load_backend_kernel.cache_clear()

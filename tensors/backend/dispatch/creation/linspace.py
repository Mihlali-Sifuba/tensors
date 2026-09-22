"""Dispatch for values built from parameters rather than transformed."""

from __future__ import annotations
from typing import TYPE_CHECKING, Any
from tensors.backend import config
from tensors.backend.config import BackendOperationUnsupportedError
from tensors.backend.loading import load_backend
from tensors.backend.storage import Storage
from tensors.backend.validation import validate_backend_residency

if TYPE_CHECKING:
    from tensors.dtype import DataType


def execute_linspace(
    start: int | float, stop: int | float, count: int, *, dtype: DataType
) -> Storage:
    """Create evenly spaced storage on the selected backend."""
    selected = config.get_backend()
    backend: Any = load_backend(selected)
    result = backend.linspace(start, stop, count, dtype=dtype)
    if result is None:
        raise BackendOperationUnsupportedError(
            f"The {selected} backend cannot execute linspace conformingly. "
            f"This computation runs on the selected backend; select another "
            f"backend to run it elsewhere."
        )
    validate_backend_residency((result,), selected)
    return result

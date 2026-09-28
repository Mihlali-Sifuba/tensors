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


def execute_full(
    shape: tuple[int, ...], fill_value: int | float, *, dtype: DataType
) -> Storage:
    """Create constant-filled storage on the selected backend."""
    selected = config.get_backend()
    backend: Any = load_backend(selected)
    result = backend.full(shape, fill_value, dtype=dtype)
    if result is None:
        raise BackendOperationUnsupportedError(
            f"The {selected} backend cannot execute full conformingly. This "
            f"computation runs on the selected backend; select another backend "
            f"to run it elsewhere."
        )
    validate_backend_residency((result,), selected)
    return result

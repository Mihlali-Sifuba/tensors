"""Dispatch for shape, layout, indexing, and representation changes."""

from __future__ import annotations
from typing import TYPE_CHECKING
from typing import Any
from tensors.backend import config
from tensors.backend.config import BackendOperationUnsupportedError
from tensors.backend.loading import load_backend
from tensors.backend.storage import Storage
from tensors.backend.validation import validate_operands, validate_result

if TYPE_CHECKING:
    from tensors.dtype import DataType
    from tensors.tensor import Tensor


def execute_cast(value: Tensor, *, dtype: DataType) -> Storage:
    """Convert dtype on the active backend without cross-backend fallback."""
    from tensors.backend.python.kernels.manipulation.cast_tensor import (
        cast_tensor as reference,
    )

    active = config.get_backend()
    validate_operands((value,), active, context="cast")
    if active == "python":
        return validate_result(
            reference(value, dtype=dtype), active, context="cast"
        )
    backend: Any = load_backend(active)
    result = backend.cast_tensor(value, dtype=dtype)
    if result is None:
        raise BackendOperationUnsupportedError(
            f"The {active} backend cannot execute cast from {value.dtype.name} "
            f"to {dtype.name} conformingly"
        )
    return validate_result(result, active, context="cast")

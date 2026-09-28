"""Strict selected-backend dispatch for vector outer products."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from tensors.backend import config
from tensors.backend.config import BackendOperationUnsupportedError
from tensors.backend.loading import load_backend
from tensors.backend.validation import validate_backend_residency

if TYPE_CHECKING:
    from tensors.backend.storage import Storage
    from tensors.dtype import DataType
    from tensors.tensor import Tensor


def execute_outer(
    left: Tensor,
    right: Tensor,
    *,
    dtype: DataType,
    output_shape: tuple[int, ...],
) -> Storage:
    """Run outer on the selected backend without fallback."""
    selected = config.get_backend()
    validate_backend_residency((left, right), selected)
    backend: Any = load_backend(selected)
    left_buffer = left._logical_storage_for(selected).buffer
    right_buffer = right._logical_storage_for(selected).buffer
    left_values = (
        left_buffer if selected == "python" else left_buffer.reshape(left.shape)
    )
    right_values = (
        right_buffer if selected == "python" else right_buffer.reshape(right.shape)
    )
    result = backend.outer(
        left_values, right_values, dtype=dtype, output_shape=output_shape
    )
    if result is None:
        raise BackendOperationUnsupportedError(
            f"The {selected} backend cannot execute outer conformingly. "
            "The outer product runs on the selected backend; select another "
            "backend to run it elsewhere."
        )
    validate_backend_residency((result,), selected)
    return result

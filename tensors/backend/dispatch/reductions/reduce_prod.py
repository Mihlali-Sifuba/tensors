"""Strict selected-backend dispatch for reduce_prod."""

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


def execute_reduce_prod(
    value: Tensor,
    axes: tuple[int, ...],
    *,
    keepdims: bool,
    dtype: DataType,
    output_shape: tuple[int, ...],
) -> Storage:
    """Run reduce_prod on the selected backend without fallback."""
    selected = config.get_backend()
    validate_backend_residency((value,), selected)
    backend: Any = load_backend(selected)
    buffer = value._logical_storage_for(selected).buffer
    values = buffer if selected == "python" else buffer.reshape(value.shape)
    result = backend.reduce_prod(
        values,
        value.shape,
        axes,
        keepdims=keepdims,
        dtype=dtype,
        output_shape=output_shape,
    )
    if result is None:
        raise BackendOperationUnsupportedError(
            f"The {selected} backend cannot execute reduce_prod conformingly. "
            "The reduction runs on the selected backend; select another backend "
            "to run it elsewhere."
        )
    validate_backend_residency((result,), selected)
    return result

"""Strict dispatch for the clip VJP."""

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


def execute_clip_gradient(
    grad: Tensor,
    value: Tensor,
    min_value: int | float | None,
    max_value: int | float | None,
    *,
    dtype: DataType,
    output_shape: tuple[int, ...],
) -> Storage:
    """Run the clip VJP on the selected backend without fallback."""
    selected = config.get_backend()
    validate_backend_residency((grad, value), selected)
    backend: Any = load_backend(selected)

    if selected == "python":
        lowered_grad = grad._data
        lowered_value = value._data
    elif selected == "numpy":
        import numpy

        native = numpy.dtype(dtype.name)
        lowered_grad = grad._logical_storage_for("numpy").buffer.reshape(grad.shape)
        lowered_value = value._logical_storage_for("numpy").buffer.reshape(value.shape)
        if lowered_grad.dtype != native:
            lowered_grad = lowered_grad.astype(native, copy=False)
        if lowered_value.dtype != native:
            lowered_value = lowered_value.astype(native, copy=False)
    else:
        import cupy

        native = cupy.dtype(dtype.name)
        lowered_grad = grad._logical_storage_for("cuda").buffer.reshape(grad.shape)
        lowered_value = value._logical_storage_for("cuda").buffer.reshape(value.shape)
        if lowered_grad.dtype != native:
            lowered_grad = lowered_grad.astype(native, copy=False)
        if lowered_value.dtype != native:
            lowered_value = lowered_value.astype(native, copy=False)

    result = backend.clip_gradient(
        lowered_grad,
        lowered_value,
        min_value,
        max_value,
        dtype=dtype,
        output_shape=output_shape,
    )
    if result is None:
        raise BackendOperationUnsupportedError(
            f"The {selected} backend cannot execute clip_gradient at dtype "
            f"{dtype.name} conformingly. The VJP runs on the selected backend; "
            f"select another backend to run it elsewhere."
        )
    validate_backend_residency((result,), selected)
    return result

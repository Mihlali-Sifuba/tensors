"""Strict dispatch for the minimum VJP."""

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


def execute_minimum_gradient(
    grad: Tensor,
    left: Tensor,
    right: Tensor,
    *,
    dtype: DataType,
    output_shape: tuple[int, ...],
    needs_input_grad: tuple[bool, ...] = (True, True),
) -> tuple[Storage | None, Storage | None]:
    """Run the minimum VJP on the selected backend without fallback."""
    selected = config.get_backend()
    validate_backend_residency((grad, left, right), selected)
    backend: Any = load_backend(selected)

    if selected == "python":
        from tensors.utils.broadcasting import broadcast_to

        lowered_grad = grad._data
        lowered_left = broadcast_to(left, output_shape)._data
        lowered_right = broadcast_to(right, output_shape)._data
    elif selected == "numpy":
        import numpy

        native = numpy.dtype(dtype.name)
        lowered_grad = grad._logical_storage_for("numpy").buffer.reshape(grad.shape)
        lowered_left = left._logical_storage_for("numpy").buffer.reshape(left.shape)
        lowered_right = right._logical_storage_for("numpy").buffer.reshape(right.shape)
        if lowered_grad.dtype != native:
            lowered_grad = lowered_grad.astype(native, copy=False)
        if lowered_left.dtype != native:
            lowered_left = lowered_left.astype(native, copy=False)
        if lowered_right.dtype != native:
            lowered_right = lowered_right.astype(native, copy=False)
    else:
        import cupy

        native = cupy.dtype(dtype.name)
        lowered_grad = grad._logical_storage_for("cuda").buffer.reshape(grad.shape)
        lowered_left = left._logical_storage_for("cuda").buffer.reshape(left.shape)
        lowered_right = right._logical_storage_for("cuda").buffer.reshape(right.shape)
        if lowered_grad.dtype != native:
            lowered_grad = lowered_grad.astype(native, copy=False)
        if lowered_left.dtype != native:
            lowered_left = lowered_left.astype(native, copy=False)
        if lowered_right.dtype != native:
            lowered_right = lowered_right.astype(native, copy=False)

    result = backend.minimum_gradient(
        lowered_grad,
        lowered_left,
        lowered_right,
        dtype=dtype,
        output_shape=output_shape,
        needs_input_grad=needs_input_grad,
    )
    if result is None:
        raise BackendOperationUnsupportedError(
            f"The {selected} backend cannot execute minimum_gradient at dtype "
            f"{dtype.name} conformingly. The VJP runs on the selected backend; "
            f"select another backend to run it elsewhere."
        )
    validate_backend_residency(
        tuple(storage for storage in result if storage is not None), selected
    )
    return result

"""Strict dispatch for the arccosh VJP."""

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


def execute_arccosh_gradient(
    grad: Tensor,
    value: Tensor,
    *,
    dtype: DataType,
    output_shape: tuple[int, ...],
) -> Storage:
    """Run the arccosh VJP on the selected backend without fallback."""
    selected = config.get_backend()
    validate_backend_residency((grad, value), selected)
    backend: Any = load_backend(selected)

    if selected == "python":
        lowered_grad = grad._data
        lowered_value = value._data
        endpoint = any(item == 1.0 for item in lowered_value)
        outside = any(item < 1.0 for item in lowered_value)
    elif selected == "numpy":
        import numpy

        lowered_grad = grad._logical_storage_for("numpy").buffer.reshape(grad.shape)
        lowered_value = value._logical_storage_for("numpy").buffer.reshape(value.shape)
        endpoint = bool(numpy.any(lowered_value == 1.0))
        outside = bool(numpy.any(lowered_value < 1.0))
    else:
        import cupy

        lowered_grad = grad._logical_storage_for("cuda").buffer.reshape(grad.shape)
        lowered_value = value._logical_storage_for("cuda").buffer.reshape(value.shape)
        endpoint = bool(cupy.any(lowered_value == 1.0))
        outside = bool(cupy.any(lowered_value < 1.0))

    if endpoint:
        raise ValueError("arccosh derivative is undefined at 1")
    if outside:
        raise ValueError("math domain error")

    result = backend.arccosh_gradient(
        lowered_grad,
        lowered_value,
        dtype=dtype,
        output_shape=output_shape,
    )
    if result is None:
        raise BackendOperationUnsupportedError(
            f"The {selected} backend cannot execute arccosh_gradient at dtype "
            f"{dtype.name} conformingly. The VJP runs on the selected backend; "
            f"select another backend to run it elsewhere."
        )
    validate_backend_residency((result,), selected)
    return result

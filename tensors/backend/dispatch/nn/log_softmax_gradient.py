"""Strict selected-backend dispatch for the log-softmax VJP."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from tensors.backend import config
from tensors.backend.config import BackendOperationUnsupportedError
from tensors.backend.loading import load_backend
from tensors.backend.validation import validate_backend_residency

if TYPE_CHECKING:
    from tensors.backend.storage import Storage
    from tensors.tensor import Tensor


def execute_log_softmax_gradient(grad: Tensor, value: Tensor, axis: int) -> Storage:
    """Run the log-softmax VJP on the selected backend without fallback."""
    selected = config.get_backend()
    validate_backend_residency((grad, value), selected)
    backend: Any = load_backend(selected)
    grad_buffer = grad._logical_storage_for(selected).buffer
    value_buffer = value._logical_storage_for(selected).buffer
    grad_values = (
        grad_buffer if selected == "python" else grad_buffer.reshape(grad.shape)
    )
    values = value_buffer if selected == "python" else value_buffer.reshape(value.shape)
    result = backend.log_softmax_gradient(
        grad_values,
        values,
        grad.shape,
        value.shape,
        axis,
        dtype=grad.dtype,
        value_dtype=value.dtype,
    )
    if result is None:
        raise BackendOperationUnsupportedError(
            f"The {selected} backend cannot execute log_softmax_gradient at "
            f"dtype {grad.dtype.name} conformingly. The VJP runs on the "
            "selected backend; select another backend to run it elsewhere."
        )
    validate_backend_residency((result,), selected)
    return result

"""Strict selected-backend dispatch for matrix-product VJPs."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from tensors.backend import config
from tensors.backend.config import BackendOperationUnsupportedError
from tensors.backend.loading import load_backend
from tensors.backend.validation import validate_backend_residency

if TYPE_CHECKING:
    from tensors.backend.storage import Storage
    from tensors.backend.types import MatmulMetadata
    from tensors.tensor import Tensor


def execute_matmul_gradient(
    grad: Tensor,
    left: Tensor,
    right: Tensor,
    *,
    metadata: MatmulMetadata,
    needs_input_grad: tuple[bool, ...] = (True, True),
) -> tuple[Storage | None, Storage | None]:
    """Run requested matmul VJPs on the selected backend without fallback."""
    selected = config.get_backend()
    validate_backend_residency((grad, left, right), selected)
    backend: Any = load_backend(selected)
    grad_buffer = grad._logical_storage_for(selected).buffer
    left_buffer = left._logical_storage_for(selected).buffer
    right_buffer = right._logical_storage_for(selected).buffer
    grad_values = (
        grad_buffer if selected == "python" else grad_buffer.reshape(grad.shape)
    )
    left_values = (
        left_buffer if selected == "python" else left_buffer.reshape(left.shape)
    )
    right_values = (
        right_buffer if selected == "python" else right_buffer.reshape(right.shape)
    )
    result = backend.matmul_gradient(
        grad_values,
        left_values,
        right_values,
        metadata=metadata,
        left_shape=left.shape,
        right_shape=right.shape,
        dtype=grad.dtype,
        needs_input_grad=needs_input_grad,
    )
    if result is None:
        raise BackendOperationUnsupportedError(
            f"The {selected} backend cannot execute matmul_gradient conformingly. "
            "The VJP runs on the selected backend; select another backend "
            "to run it elsewhere."
        )
    validate_backend_residency(
        (storage for storage in result if storage is not None), selected
    )
    return result

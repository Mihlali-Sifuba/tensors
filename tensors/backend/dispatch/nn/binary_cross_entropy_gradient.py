"""Strict selected-backend dispatch for the binary cross-entropy VJP."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from tensors.backend import config
from tensors.backend.config import BackendOperationUnsupportedError
from tensors.backend.loading import load_backend
from tensors.backend.validation import validate_backend_residency

if TYPE_CHECKING:
    from tensors.backend.storage import Storage
    from tensors.backend.types import LossReduction
    from tensors.tensor import Tensor


def execute_binary_cross_entropy_gradient(
    grad: Tensor,
    prediction: Tensor,
    target: Tensor,
    *,
    from_logits: bool,
    reduction: LossReduction,
    needs_input_grad: tuple[bool, ...] = (True, True),
) -> tuple[Storage | None, Storage | None]:
    """Run requested binary cross-entropy VJPs without fallback."""
    selected = config.get_backend()
    validate_backend_residency((grad, prediction, target), selected)
    backend: Any = load_backend(selected)
    grad_buffer = grad._logical_storage_for(selected).buffer
    prediction_buffer = prediction._logical_storage_for(selected).buffer
    target_buffer = target._logical_storage_for(selected).buffer
    grad_values = (
        grad_buffer if selected == "python" else grad_buffer.reshape(grad.shape)
    )
    prediction_values = (
        prediction_buffer
        if selected == "python"
        else prediction_buffer.reshape(prediction.shape)
    )
    target_values = (
        target_buffer if selected == "python" else target_buffer.reshape(target.shape)
    )
    result = backend.binary_cross_entropy_gradient(
        grad_values,
        prediction_values,
        target_values,
        grad.shape,
        prediction.shape,
        target.shape,
        from_logits=from_logits,
        reduction=reduction,
        dtype=grad.dtype,
        needs_input_grad=needs_input_grad,
    )
    if result is None:
        raise BackendOperationUnsupportedError(
            f"The {selected} backend cannot execute binary_cross_entropy_gradient "
            f"at dtype {grad.dtype.name} conformingly. The VJP runs on the "
            "selected backend; select another backend to run it elsewhere."
        )
    validate_backend_residency(
        (storage for storage in result if storage is not None), selected
    )
    return result

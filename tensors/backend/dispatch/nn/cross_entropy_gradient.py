"""Strict selected-backend dispatch for the cross-entropy VJP."""

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


def execute_cross_entropy_gradient(
    grad: Tensor,
    logits: Tensor,
    targets: Tensor,
    axis: int,
    *,
    reduction: LossReduction,
    needs_input_grad: tuple[bool, ...] = (True, True),
) -> tuple[Storage | None, Storage | None]:
    """Run requested dense cross-entropy VJPs without fallback."""
    selected = config.get_backend()
    validate_backend_residency((grad, logits, targets), selected)
    backend: Any = load_backend(selected)
    grad_buffer = grad._logical_storage_for(selected).buffer
    logits_buffer = logits._logical_storage_for(selected).buffer
    targets_buffer = targets._logical_storage_for(selected).buffer
    grad_values = (
        grad_buffer if selected == "python" else grad_buffer.reshape(grad.shape)
    )
    logits_values = (
        logits_buffer if selected == "python" else logits_buffer.reshape(logits.shape)
    )
    target_values = (
        targets_buffer
        if selected == "python"
        else targets_buffer.reshape(targets.shape)
    )
    result = backend.cross_entropy_gradient(
        grad_values,
        logits_values,
        target_values,
        grad.shape,
        logits.shape,
        targets.shape,
        axis,
        reduction=reduction,
        dtype=grad.dtype,
        logits_dtype=logits.dtype,
        needs_input_grad=needs_input_grad,
    )
    if result is None:
        raise BackendOperationUnsupportedError(
            f"The {selected} backend cannot execute cross_entropy_gradient at "
            f"dtype {grad.dtype.name} conformingly. The VJP runs on the selected "
            "backend; select another backend to run it elsewhere."
        )
    validate_backend_residency(
        (storage for storage in result if storage is not None), selected
    )
    return result

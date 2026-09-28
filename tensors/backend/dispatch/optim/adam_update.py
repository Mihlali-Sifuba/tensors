"""Dispatch for optimizer updates."""

from __future__ import annotations
from typing import TYPE_CHECKING, Any

from tensors.backend import config
from tensors.backend.config import BackendOperationUnsupportedError
from tensors.backend.loading import load_backend
from tensors.backend.validation import validate_backend_residency

if TYPE_CHECKING:
    from tensors.backend.storage import Storage
    from tensors.tensor import Tensor


def execute_adam_update(
    parameter: Tensor,
    gradient: Tensor,
    moment: Tensor,
    scale: Tensor,
    scaled: Tensor,
    *,
    beta1: float,
    beta2: float,
    learning_rate: float,
    epsilon: float,
    first_correction: float,
    second_correction: float,
) -> tuple[Storage, Storage, Storage, Storage, Storage]:
    """Run one Adam update on the selected backend without fallback."""
    tensors = (parameter, gradient, moment, scale, scaled)
    selected = config.get_backend()
    validate_backend_residency(tensors, selected)
    backend: Any = load_backend(selected)
    buffers = tuple(tensor._logical_storage_for(selected).buffer for tensor in tensors)
    values = tuple(
        buffer if selected == "python" else buffer.reshape(tensor.shape)
        for tensor, buffer in zip(tensors, buffers)
    )
    result = backend.adam_update(
        *values,
        beta1=beta1,
        beta2=beta2,
        learning_rate=learning_rate,
        epsilon=epsilon,
        first_correction=first_correction,
        second_correction=second_correction,
        dtype=parameter.dtype,
        shape=tuple(parameter.shape),
    )
    if result is None:
        raise BackendOperationUnsupportedError(
            f"The {selected} backend cannot execute adam_update at dtype "
            f"{parameter.dtype.name} conformingly. Optimizer updates run on "
            "the selected backend; select another backend to run it elsewhere."
        )
    validate_backend_residency(result, selected)
    return result

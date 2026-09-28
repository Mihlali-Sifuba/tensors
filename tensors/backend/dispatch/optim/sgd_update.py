"""Strict selected-backend dispatch for one SGD update."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from tensors.backend import config
from tensors.backend.config import BackendOperationUnsupportedError
from tensors.backend.loading import load_backend
from tensors.backend.validation import validate_backend_residency

if TYPE_CHECKING:
    from tensors.backend.storage import Storage
    from tensors.tensor import Tensor


def execute_sgd_update(
    parameter: Tensor, gradient: Tensor, learning_rate: float
) -> Storage:
    """Run one SGD update on the selected backend without fallback."""
    selected = config.get_backend()
    validate_backend_residency((parameter, gradient), selected)
    backend: Any = load_backend(selected)
    parameter_buffer = parameter._logical_storage_for(selected).buffer
    gradient_buffer = gradient._logical_storage_for(selected).buffer
    parameter_values = (
        parameter_buffer
        if selected == "python"
        else parameter_buffer.reshape(parameter.shape)
    )
    gradient_values = (
        gradient_buffer
        if selected == "python"
        else gradient_buffer.reshape(gradient.shape)
    )
    result = backend.sgd_update(
        parameter_values,
        gradient_values,
        learning_rate,
        dtype=parameter.dtype,
        shape=tuple(parameter.shape),
    )
    if result is None:
        raise BackendOperationUnsupportedError(
            f"The {selected} backend cannot execute sgd_update at dtype "
            f"{parameter.dtype.name} conformingly. Optimizer updates run on "
            "the selected backend; select another backend to run it elsewhere."
        )
    validate_backend_residency((result,), selected)
    return result

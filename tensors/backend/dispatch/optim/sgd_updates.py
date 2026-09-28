"""Dispatch for optimizer updates."""

from __future__ import annotations
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from tensors.backend import config
from tensors.backend.loading import load_backend
from tensors.backend.validation import validate_backend_residency

if TYPE_CHECKING:
    from tensors.backend.storage import Storage
    from tensors.tensor import Tensor


def execute_sgd_updates(
    parameters: Sequence[Tensor], gradients: Sequence[Tensor], learning_rate: float
) -> tuple[Storage, ...] | None:
    """Try one native grouped SGD update on the selected backend."""
    count = len(parameters)
    if len(gradients) != count:
        raise ValueError("grouped SGD parameters and gradients must have equal length")
    selected = config.get_backend()
    validate_backend_residency((*parameters, *gradients), selected)
    if count < 2 or selected == "python":
        return None
    backend: Any = load_backend(selected)
    parameter_values = tuple(
        parameter._logical_storage_for(selected).buffer.reshape(parameter.shape)
        for parameter in parameters
    )
    gradient_values = tuple(
        gradient._logical_storage_for(selected).buffer.reshape(gradient.shape)
        for gradient in gradients
    )
    result = backend.sgd_updates(
        parameter_values,
        gradient_values,
        learning_rate,
        dtypes=tuple(parameter.dtype for parameter in parameters),
        shapes=tuple(tuple(parameter.shape) for parameter in parameters),
    )
    if result is None:
        return None
    if len(result) != count:
        raise RuntimeError("Grouped SGD kernel returned an unexpected result count")
    validate_backend_residency(result, selected)
    return result

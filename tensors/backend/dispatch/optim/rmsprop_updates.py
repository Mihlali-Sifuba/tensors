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


def execute_rmsprop_updates(
    parameters: Sequence[Tensor],
    gradients: Sequence[Tensor],
    scales: Sequence[Tensor],
    scaled_values: Sequence[Tensor],
    *,
    rho: float,
    learning_rate: float,
    epsilon: float,
) -> tuple[tuple[Storage, ...], ...] | None:
    """Update several compatible RMSprop states in one native array batch."""
    count = len(parameters)
    if any(len(items) != count for items in (gradients, scales, scaled_values)):
        raise ValueError("grouped RMSprop input sequences must have equal length")
    selected = config.get_backend()
    tensors = (*parameters, *gradients, *scales, *scaled_values)
    validate_backend_residency(tensors, selected)
    if count < 2 or selected == "python":
        return None
    backend: Any = load_backend(selected)

    parameter_values = tuple(
        value._logical_storage_for(selected).buffer.reshape(value.shape)
        for value in parameters
    )
    gradient_values = tuple(
        value._logical_storage_for(selected).buffer.reshape(value.shape)
        for value in gradients
    )
    scale_values = tuple(
        value._logical_storage_for(selected).buffer.reshape(value.shape)
        for value in scales
    )
    normalized_values = tuple(
        value._logical_storage_for(selected).buffer.reshape(value.shape)
        for value in scaled_values
    )
    result = backend.rmsprop_updates(
        parameter_values,
        gradient_values,
        scale_values,
        normalized_values,
        rho=rho,
        learning_rate=learning_rate,
        epsilon=epsilon,
        dtypes=tuple(parameter.dtype for parameter in parameters),
        shapes=tuple(tuple(parameter.shape) for parameter in parameters),
    )
    if result is None:
        return None
    if len(result) != 3 or any(len(group) != count for group in result):
        raise RuntimeError("Grouped RMSprop kernel returned unexpected result counts")
    validate_backend_residency(
        tuple(storage for group in result for storage in group), selected
    )
    return result

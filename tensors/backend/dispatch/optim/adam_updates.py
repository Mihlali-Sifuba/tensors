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


def execute_adam_updates(
    parameters: Sequence[Tensor],
    gradients: Sequence[Tensor],
    moments: Sequence[Tensor],
    scales: Sequence[Tensor],
    scaled_values: Sequence[Tensor],
    *,
    beta1: float,
    beta2: float,
    learning_rate: float,
    epsilon: float,
    first_corrections: Sequence[float],
    second_corrections: Sequence[float],
) -> tuple[tuple[Storage, ...], ...] | None:
    """Update several compatible Adam states in one native array batch."""
    count = len(parameters)
    sequences = (
        gradients,
        moments,
        scales,
        scaled_values,
        first_corrections,
        second_corrections,
    )
    if any(len(items) != count for items in sequences):
        raise ValueError("grouped Adam input sequences must have equal length")
    selected = config.get_backend()
    tensors = (*parameters, *gradients, *moments, *scales, *scaled_values)
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
    moment_values = tuple(
        value._logical_storage_for(selected).buffer.reshape(value.shape)
        for value in moments
    )
    scale_values = tuple(
        value._logical_storage_for(selected).buffer.reshape(value.shape)
        for value in scales
    )
    normalized_values = tuple(
        value._logical_storage_for(selected).buffer.reshape(value.shape)
        for value in scaled_values
    )
    result = backend.adam_updates(
        parameter_values,
        gradient_values,
        moment_values,
        scale_values,
        normalized_values,
        beta1=beta1,
        beta2=beta2,
        learning_rate=learning_rate,
        epsilon=epsilon,
        first_corrections=first_corrections,
        second_corrections=second_corrections,
        dtypes=tuple(parameter.dtype for parameter in parameters),
        shapes=tuple(tuple(parameter.shape) for parameter in parameters),
    )
    if result is None:
        return None
    if len(result) != 5 or any(len(group) != count for group in result):
        raise RuntimeError("Grouped Adam kernel returned unexpected result counts")
    validate_backend_residency(
        tuple(storage for group in result for storage in group), selected
    )
    return result

"""NumPy implementation of the batched RMSprop parameter update."""

from __future__ import annotations
import numpy
from collections.abc import Sequence
from typing import TYPE_CHECKING
from typing import Any, cast
from tensors.backend.storage import Storage
from tensors.backend.numpy.conversion import _errstate
from tensors.backend.numpy.kernels.optim.batching import _optimizer_batch_partitions
from tensors.backend.numpy.kernels.optim.batching import _optimizer_batch_values
from tensors.backend.numpy.kernels.optim.batching import _optimizer_partition
from tensors.backend.numpy.kernels.optim.batching import _split_optimizer_storage

if TYPE_CHECKING:
    from tensors.tensor import Tensor


def rmsprop_updates(
    parameters: Sequence[Tensor],
    gradients: Sequence[Tensor],
    scales: Sequence[Tensor],
    scaled_values: Sequence[Tensor],
    *,
    rho: float,
    learning_rate: float,
    epsilon: float,
) -> tuple[tuple[Storage, ...], ...] | None:
    """Apply RMSprop to several parameters with one group of array operations."""
    groups = (parameters, gradients, scales, scaled_values)
    partitions = _optimizer_batch_partitions(
        parameters, gradients, scales, scaled_values
    )
    if partitions is None:
        return None
    if len(partitions) > 1:
        merged: list[list[Storage | None]] = [
            [None] * len(parameters) for _ in range(3)
        ]
        for indices in partitions:
            result = rmsprop_updates(
                *(_optimizer_partition(group, indices) for group in groups),
                rho=rho,
                learning_rate=learning_rate,
                epsilon=epsilon,
            )
            if result is None:
                return None
            for merged_group, result_group in zip(merged, result):
                for index, storage in zip(indices, result_group):
                    merged_group[index] = storage
        return tuple(
            (tuple((cast(Storage, storage) for storage in group)) for group in merged)
        )
    optional_arrays = tuple(
        (
            _optimizer_batch_values(group, slot=f"input:{index}")
            for index, group in enumerate(groups)
        )
    )
    if any((array is None for array in optional_arrays)):
        return None
    arrays = cast("tuple[Any, ...]", optional_arrays)
    parameter_values, gradient_values, scale_values, normalized_values = arrays
    with _errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore"):
        new_scales = numpy.maximum(scale_values, numpy.abs(gradient_values))
        safe_scales = numpy.where(new_scales == 0.0, 1.0, new_scales)
        previous_ratio = scale_values / safe_scales
        gradient_ratio = numpy.abs(gradient_values) / safe_scales
        new_scaled = (
            rho * normalized_values * previous_ratio * previous_ratio
            + (1.0 - rho) * gradient_ratio * gradient_ratio
        )
        new_scaled = numpy.where(new_scales == 0.0, 0.0, new_scaled)
        root_moment = new_scales * numpy.sqrt(new_scaled)
        parameter_result = parameter_values - learning_rate * gradient_values / (
            root_moment + epsilon
        )
    finite_inputs = numpy.asarray(True)
    for array in arrays:
        finite_inputs &= numpy.all(numpy.isfinite(array))
    valid = (
        finite_inputs
        & numpy.all(numpy.isfinite(new_scales))
        & numpy.all(numpy.isfinite(new_scaled))
        & numpy.all(numpy.isfinite(parameter_result))
    )
    if not bool(valid):
        return None
    results = (
        _split_optimizer_storage(parameter_result, parameters),
        _split_optimizer_storage(new_scales, gradients),
        _split_optimizer_storage(new_scaled, gradients),
    )
    if any((result is None for result in results)):
        return None
    return cast("tuple[tuple[Storage, ...], ...]", results)

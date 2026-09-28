"""CuPy implementation of the batched Adam parameter update."""

from __future__ import annotations
import cupy
from collections.abc import Sequence
from typing import TYPE_CHECKING
from typing import Any, cast
from tensors.backend.storage import Storage
from tensors.backend.cuda.kernels.optim.batching import _optimizer_batch_partitions
from tensors.backend.cuda.kernels.optim.batching import _optimizer_batch_values
from tensors.backend.cuda.kernels.optim.batching import _optimizer_invalid_flag
from tensors.backend.cuda.kernels.optim.batching import _optimizer_partition
from tensors.backend.cuda.kernels.optim.batching import _optimizer_scalar_batch
from tensors.backend.cuda.kernels.optim.batching import _split_optimizer_storage
from tensors.backend.cuda.kernels.optim.batch_kernels import _cuda_adam_batch_kernel

if TYPE_CHECKING:
    from tensors.dtype import DataType


def adam_updates(
    parameters: Sequence[Any],
    gradients: Sequence[Any],
    moments: Sequence[Any],
    scales: Sequence[Any],
    scaled_values: Sequence[Any],
    *,
    beta1: float,
    beta2: float,
    learning_rate: float,
    epsilon: float,
    first_corrections: Sequence[float],
    second_corrections: Sequence[float],
    dtypes: Sequence[DataType],
    shapes: Sequence[tuple[int, ...]],
) -> tuple[tuple[Storage, ...], ...] | None:
    """Apply Adam to several parameters with one group of array operations."""
    groups = (parameters, gradients, moments, scales, scaled_values)
    partitions = _optimizer_batch_partitions(
        dtypes, shapes, parameters, gradients, moments, scales, scaled_values
    )
    if partitions is None:
        return None
    if len(partitions) > 1:
        merged: list[list[Storage | None]] = [
            [None] * len(parameters) for _ in range(5)
        ]
        for indices in partitions:
            result = adam_updates(
                *(_optimizer_partition(group, indices) for group in groups),
                beta1=beta1,
                beta2=beta2,
                learning_rate=learning_rate,
                epsilon=epsilon,
                first_corrections=_optimizer_partition(first_corrections, indices),
                second_corrections=_optimizer_partition(second_corrections, indices),
                dtypes=_optimizer_partition(dtypes, indices),
                shapes=_optimizer_partition(shapes, indices),
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
    (
        parameter_values,
        gradient_values,
        moment_values,
        scale_values,
        normalized_values,
    ) = arrays
    first = _optimizer_scalar_batch(first_corrections, shapes)
    second = _optimizer_scalar_batch(second_corrections, shapes)
    parameter_result = cupy.empty_like(parameter_values)
    new_moments = cupy.empty_like(moment_values)
    visible = cupy.empty_like(moment_values)
    new_scales = cupy.empty_like(scale_values)
    new_scaled = cupy.empty_like(normalized_values)
    invalid = _optimizer_invalid_flag()
    _cuda_adam_batch_kernel()(
        parameter_values,
        gradient_values,
        moment_values,
        scale_values,
        normalized_values,
        beta1,
        beta2,
        learning_rate,
        epsilon,
        first,
        second,
        parameter_result,
        new_moments,
        visible,
        new_scales,
        new_scaled,
        invalid,
    )
    if bool(invalid[0]):
        return None
    results = (
        _split_optimizer_storage(parameter_result, dtypes, shapes),
        _split_optimizer_storage(new_moments, dtypes, shapes),
        _split_optimizer_storage(visible, dtypes, shapes),
        _split_optimizer_storage(new_scales, dtypes, shapes),
        _split_optimizer_storage(new_scaled, dtypes, shapes),
    )
    if any((result is None for result in results)):
        return None
    return cast("tuple[tuple[Storage, ...], ...]", results)

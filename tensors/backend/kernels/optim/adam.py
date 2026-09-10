"""Adam updates."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, TYPE_CHECKING, cast

from ...storage import Storage
from ..core import (
    _array_kind,
    _errstate,
    _finite_operands,
    _numpy,
    _storage,
    _view,
)
from .batching import (
    _optimizer_batch_partitions,
    _optimizer_batch_values,
    _optimizer_invalid_flag,
    _optimizer_partition,
    _optimizer_scalar_batch,
    _split_optimizer_storage,
)
from .cuda import _cuda_adam_batch_kernel

if TYPE_CHECKING:
    from ....tensor import Tensor

def adam_update(
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
) -> tuple[Storage, Storage, Storage, Storage, Storage] | None:
    """Apply one fused Adam update on finite optimizer state."""
    numpy = _numpy()
    tensors = (parameter, gradient, moment, scale, scaled)
    values = [
        _view(item, numpy).astype(numpy.float64, copy=False)
        for item in tensors
    ]
    parameter_values, gradients, moments, scales, scaled_values = values
    if not _finite_operands(*values, numpy=numpy):
        return None
    left_term = beta1 * moments
    right_term = (1.0 - beta1) * gradients
    with _errstate(numpy, over="ignore", under="ignore", invalid="ignore"):
        new_moments = left_term + right_term
        new_scales = numpy.maximum(scales, numpy.abs(gradients))
        safe_scales = numpy.where(new_scales == 0.0, 1.0, new_scales)
        previous_ratio = scales / safe_scales
        gradient_ratio = numpy.abs(gradients) / safe_scales
        new_scaled = (
            beta2 * scaled_values * previous_ratio * previous_ratio
            + (1.0 - beta2) * gradient_ratio * gradient_ratio
        )
        new_scaled = numpy.where(new_scales == 0.0, 0.0, new_scaled)
        root_correction = numpy.sqrt(second_correction)
        root_moment = new_scales * numpy.sqrt(new_scaled)
        denominator = first_correction * (
            root_moment + epsilon * root_correction
        )
        ratio = new_moments * root_correction / denominator
        parameter_result = parameter_values - learning_rate * ratio
        visible = new_scales * new_scales * new_scaled
    if not _finite_operands(
        new_moments,
        new_scales,
        new_scaled,
        parameter_result,
        numpy=numpy,
    ):
        return None
    specifications = (
        (parameter_result, parameter.dtype),
        (new_moments, gradient.dtype),
        (visible, gradient.dtype),
        (new_scales, gradient.dtype),
        (new_scaled, gradient.dtype),
    )
    storages = tuple(
        _storage(
            result,
            dtype=dtype,
            output_shape=gradient.shape,
            numpy=numpy,
        )
        for result, dtype in specifications
    )
    if any(storage is None for storage in storages):
        return None
    return cast(
        "tuple[Storage, Storage, Storage, Storage, Storage]",
        storages,
    )

def adam_updates(
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
    """Apply Adam to several parameters with one group of array operations."""
    groups = (parameters, gradients, moments, scales, scaled_values)
    partitions = _optimizer_batch_partitions(
        parameters,
        gradients,
        moments,
        scales,
        scaled_values,
    )
    if partitions is None:
        return None
    if len(partitions) > 1:
        merged: list[list[Storage | None]] = [
            [None] * len(parameters) for _ in range(5)
        ]
        for indices in partitions:
            result = adam_updates(
                *(
                    _optimizer_partition(group, indices)
                    for group in groups
                ),
                beta1=beta1,
                beta2=beta2,
                learning_rate=learning_rate,
                epsilon=epsilon,
                first_corrections=_optimizer_partition(
                    first_corrections,
                    indices,
                ),
                second_corrections=_optimizer_partition(
                    second_corrections,
                    indices,
                ),
            )
            if result is None:
                return None
            for merged_group, result_group in zip(merged, result):
                for index, storage in zip(indices, result_group):
                    merged_group[index] = storage
        return tuple(
            tuple(cast(Storage, storage) for storage in group)
            for group in merged
        )
    numpy = _numpy()
    optional_arrays = tuple(
        _optimizer_batch_values(group, numpy, slot=f"input:{index}")
        for index, group in enumerate(groups)
    )
    if any(array is None for array in optional_arrays):
        return None
    arrays = cast("tuple[Any, ...]", optional_arrays)
    (
        parameter_values,
        gradient_values,
        moment_values,
        scale_values,
        normalized_values,
    ) = arrays
    first = _optimizer_scalar_batch(first_corrections, parameters, numpy)
    second = _optimizer_scalar_batch(second_corrections, parameters, numpy)
    if _array_kind(numpy) == "cuda":
        parameter_result = numpy.empty_like(parameter_values)
        new_moments = numpy.empty_like(moment_values)
        visible = numpy.empty_like(moment_values)
        new_scales = numpy.empty_like(scale_values)
        new_scaled = numpy.empty_like(normalized_values)
        invalid = _optimizer_invalid_flag(numpy)
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
    else:
        left_term = beta1 * moment_values
        right_term = (1.0 - beta1) * gradient_values
        with _errstate(
            numpy,
            divide="ignore",
            over="ignore",
            under="ignore",
            invalid="ignore",
        ):
            new_moments = left_term + right_term
            new_scales = numpy.maximum(
                scale_values,
                numpy.abs(gradient_values),
            )
            safe_scales = numpy.where(new_scales == 0.0, 1.0, new_scales)
            previous_ratio = scale_values / safe_scales
            gradient_ratio = numpy.abs(gradient_values) / safe_scales
            new_scaled = (
                beta2 * normalized_values * previous_ratio * previous_ratio
                + (1.0 - beta2) * gradient_ratio * gradient_ratio
            )
            new_scaled = numpy.where(new_scales == 0.0, 0.0, new_scaled)
            root_correction = numpy.sqrt(second)
            root_moment = new_scales * numpy.sqrt(new_scaled)
            denominator = first * (
                root_moment + epsilon * root_correction
            )
            ratio = new_moments * root_correction / denominator
            parameter_result = parameter_values - learning_rate * ratio
            visible = new_scales * new_scales * new_scaled
        finite_inputs = numpy.asarray(True)
        for array in arrays:
            finite_inputs &= numpy.all(numpy.isfinite(array))
        valid = (
            finite_inputs
            & numpy.all(numpy.isfinite(new_moments))
            & numpy.all(numpy.isfinite(new_scales))
            & numpy.all(numpy.isfinite(new_scaled))
            & numpy.all(numpy.isfinite(parameter_result))
        )
        if not bool(valid):
            return None
    results = (
        _split_optimizer_storage(parameter_result, parameters, numpy),
        _split_optimizer_storage(new_moments, gradients, numpy),
        _split_optimizer_storage(visible, gradients, numpy),
        _split_optimizer_storage(new_scales, gradients, numpy),
        _split_optimizer_storage(new_scaled, gradients, numpy),
    )
    if any(result is None for result in results):
        return None
    return cast("tuple[tuple[Storage, ...], ...]", results)

"""Optimizer kernels and their batched CUDA workspaces."""

from __future__ import annotations

import importlib
import threading
from collections.abc import Sequence
from functools import lru_cache
from typing import Any, TYPE_CHECKING, cast

from ..storage import CudaStorage, NumPyStorage, Storage
from .core import _array_kind, _errstate, _finite_operands, _numpy, _storage, _view

if TYPE_CHECKING:
    from ...tensor import Tensor

_optimizer_workspace = threading.local()

def _optimizer_workspace_buffer(
    numpy: Any,
    *,
    slot: str,
    size: int,
    dtype: Any,
) -> Any:
    """Return one thread-local temporary buffer for optimizer execution."""
    buffers = getattr(_optimizer_workspace, "buffers", None)
    if buffers is None:
        buffers = {}
        _optimizer_workspace.buffers = buffers
    provider_dtype = numpy.dtype(dtype)
    key = (_array_kind(numpy), slot, provider_dtype.str)
    buffer = buffers.get(key)
    if buffer is None or buffer.size != size:
        buffer = numpy.empty((size,), dtype=provider_dtype)
        buffers[key] = buffer
    return buffer

def _optimizer_batch_values(
    tensors: Sequence[Tensor],
    numpy: Any,
    *,
    slot: str,
) -> Any | None:
    """Pack compatible optimizer tensors into reusable native storage."""
    if not tensors:
        return None
    dtype = tensors[0].dtype
    if any(tensor.dtype != dtype for tensor in tensors):
        return None
    arrays = tuple(
        _view(tensor, numpy).astype(numpy.float64, copy=False).reshape(-1)
        for tensor in tensors
    )
    buffer = _optimizer_workspace_buffer(
        numpy,
        slot=slot,
        size=sum(tensor.size for tensor in tensors),
        dtype=numpy.float64,
    )
    numpy.concatenate(arrays, out=buffer)
    return buffer

def _optimizer_batch_partitions(
    *groups: Sequence[Tensor],
) -> tuple[tuple[int, ...], ...] | None:
    """Group structurally compatible optimizer records by dtype."""
    if not groups or not groups[0]:
        return None
    count = len(groups[0])
    if any(len(group) != count for group in groups):
        return None
    partitions: dict[Any, list[int]] = {}
    for index, tensors in enumerate(zip(*groups)):
        shape = tensors[0].shape
        dtype = tensors[0].dtype
        if any(
            tensor.shape != shape or tensor.dtype != dtype
            for tensor in tensors
        ):
            return None
        partitions.setdefault(dtype, []).append(index)
    return tuple(tuple(indices) for indices in partitions.values())

def _optimizer_partition(
    values: Sequence[Any],
    indices: tuple[int, ...],
) -> tuple[Any, ...]:
    """Select one optimizer batch partition while preserving record order."""
    return tuple(values[index] for index in indices)

def _optimizer_invalid_flag(numpy: Any) -> Any:
    """Return a cleared scalar device flag for fused optimizer validation."""
    flag = _optimizer_workspace_buffer(
        numpy,
        slot="invalid",
        size=1,
        dtype=numpy.uint32,
    )
    flag.fill(0)
    return flag

def _split_optimizer_storage(
    result: Any,
    references: Sequence[Tensor],
    numpy: Any,
) -> tuple[Storage, ...] | None:
    """Retain slices of one batched result without copying them again."""
    if not references:
        return ()
    dtype = references[0].dtype
    total = sum(reference.size for reference in references)
    storage = _storage(
        result,
        dtype=dtype,
        output_shape=(total,),
        numpy=numpy,
    )
    if storage is None:
        return None
    storages: list[Storage] = []
    offset = 0
    for reference in references:
        end = offset + reference.size
        buffer = storage.buffer[offset:end]
        if storage.kind == "cuda":
            storages.append(CudaStorage(buffer, dtype))
        else:
            storages.append(NumPyStorage(buffer, dtype))
        offset = end
    return tuple(storages)

def sgd_update(
    parameter: Tensor,
    gradient: Tensor,
    learning_rate: float,
) -> Storage | None:
    """Apply one fused SGD update."""
    numpy = _numpy()
    values = _view(parameter, numpy).astype(numpy.float64, copy=False)
    gradients = _view(gradient, numpy).astype(numpy.float64, copy=False)
    if not _finite_operands(values, gradients, numpy=numpy):
        return None
    with _errstate(numpy, over="ignore", under="ignore", invalid="ignore"):
        result = values - learning_rate * gradients
    if not bool(numpy.all(numpy.isfinite(result))):
        return None
    return _storage(
        result,
        dtype=parameter.dtype,
        output_shape=parameter.shape,
        numpy=numpy,
    )

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

def rmsprop_update(
    parameter: Tensor,
    gradient: Tensor,
    scale: Tensor,
    scaled: Tensor,
    *,
    rho: float,
    learning_rate: float,
    epsilon: float,
) -> tuple[Storage, Storage, Storage] | None:
    """Apply one fused RMSprop update on finite optimizer state."""
    numpy = _numpy()
    tensors = (parameter, gradient, scale, scaled)
    values = [
        _view(item, numpy).astype(numpy.float64, copy=False)
        for item in tensors
    ]
    parameter_values, gradients, scales, scaled_values = values
    if not _finite_operands(*values, numpy=numpy):
        return None
    with _errstate(numpy, over="ignore", under="ignore", invalid="ignore"):
        new_scales = numpy.maximum(scales, numpy.abs(gradients))
        safe_scales = numpy.where(new_scales == 0.0, 1.0, new_scales)
        previous_ratio = scales / safe_scales
        gradient_ratio = numpy.abs(gradients) / safe_scales
        new_scaled = (
            rho * scaled_values * previous_ratio * previous_ratio
            + (1.0 - rho) * gradient_ratio * gradient_ratio
        )
        new_scaled = numpy.where(new_scales == 0.0, 0.0, new_scaled)
        root_moment = new_scales * numpy.sqrt(new_scaled)
        parameter_result = parameter_values - (
            learning_rate * gradients / (root_moment + epsilon)
        )
    if not _finite_operands(
        new_scales,
        new_scaled,
        parameter_result,
        numpy=numpy,
    ):
        return None
    specifications = (
        (parameter_result, parameter.dtype),
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
        "tuple[Storage, Storage, Storage]",
        storages,
    )

def sgd_updates(
    parameters: Sequence[Tensor],
    gradients: Sequence[Tensor],
    learning_rate: float,
) -> tuple[Storage, ...] | None:
    """Apply one native SGD update to several compatible parameters."""
    partitions = _optimizer_batch_partitions(parameters, gradients)
    if partitions is None:
        return None
    if len(partitions) > 1:
        merged: list[Storage | None] = [None] * len(parameters)
        for indices in partitions:
            result = sgd_updates(
                _optimizer_partition(parameters, indices),
                _optimizer_partition(gradients, indices),
                learning_rate,
            )
            if result is None:
                return None
            for index, storage in zip(indices, result):
                merged[index] = storage
        return tuple(cast(Storage, storage) for storage in merged)
    numpy = _numpy()
    values = _optimizer_batch_values(parameters, numpy, slot="input:0")
    gradient_values = _optimizer_batch_values(
        gradients,
        numpy,
        slot="input:1",
    )
    if values is None or gradient_values is None:
        return None
    if _array_kind(numpy) == "cuda":
        result = numpy.empty_like(values)
        invalid = _optimizer_invalid_flag(numpy)
        _cuda_sgd_batch_kernel()(
            values,
            gradient_values,
            learning_rate,
            result,
            invalid,
        )
        if bool(invalid[0]):
            return None
    else:
        with _errstate(numpy, over="ignore", under="ignore", invalid="ignore"):
            result = values - learning_rate * gradient_values
        valid = (
            numpy.all(numpy.isfinite(values))
            & numpy.all(numpy.isfinite(gradient_values))
            & numpy.all(numpy.isfinite(result))
        )
        if not bool(valid):
            return None
    return _split_optimizer_storage(result, parameters, numpy)

@lru_cache(maxsize=1)
def _cuda_sgd_batch_kernel() -> Any:
    """Return a fused CUDA SGD update and validation kernel."""
    cupy = importlib.import_module("cupy")
    return cupy.ElementwiseKernel(
        "float64 parameter, float64 gradient, float64 learning_rate",
        "float64 updated, raw uint32 invalid",
        """
        updated = parameter - learning_rate * gradient;
        if (
            !isfinite(parameter)
            || !isfinite(gradient)
            || !isfinite(updated)
        ) atomicExch(&invalid[0], 1U);
        """,
        "tensors_sgd_batch",
    )

@lru_cache(maxsize=1)
def _cuda_adam_batch_kernel() -> Any:
    """Return a fused CUDA Adam update and validation kernel."""
    cupy = importlib.import_module("cupy")
    return cupy.ElementwiseKernel(
        """
        float64 parameter, float64 gradient, float64 moment,
        float64 scale, float64 normalized, float64 beta1, float64 beta2,
        float64 learning_rate, float64 epsilon,
        float64 first_correction, float64 second_correction
        """,
        """
        float64 updated, float64 new_moment, float64 visible,
        float64 new_scale, float64 new_normalized, raw uint32 invalid
        """,
        """
        double left = beta1 * moment;
        double right = (1.0 - beta1) * gradient;
        new_moment = left + right;
        new_scale = fmax(scale, fabs(gradient));
        double safe_scale = new_scale == 0.0 ? 1.0 : new_scale;
        double previous_ratio = scale / safe_scale;
        double gradient_ratio = fabs(gradient) / safe_scale;
        new_normalized = (
            beta2 * normalized * previous_ratio * previous_ratio
            + (1.0 - beta2) * gradient_ratio * gradient_ratio
        );
        if (new_scale == 0.0) new_normalized = 0.0;
        double root_correction = sqrt(second_correction);
        double root_moment = new_scale * sqrt(new_normalized);
        double denominator = first_correction * (
            root_moment + epsilon * root_correction
        );
        double ratio = new_moment * root_correction / denominator;
        updated = parameter - learning_rate * ratio;
        visible = new_scale * new_scale * new_normalized;
        if (
            !isfinite(parameter)
            || !isfinite(gradient)
            || !isfinite(moment)
            || !isfinite(scale)
            || !isfinite(normalized)
            || !isfinite(updated)
            || !isfinite(new_moment)
            || !isfinite(new_scale)
            || !isfinite(new_normalized)
        ) atomicExch(&invalid[0], 1U);
        """,
        "tensors_adam_batch",
    )

@lru_cache(maxsize=1)
def _cuda_rmsprop_batch_kernel() -> Any:
    """Return a fused CUDA RMSprop update and validation kernel."""
    cupy = importlib.import_module("cupy")
    return cupy.ElementwiseKernel(
        """
        float64 parameter, float64 gradient, float64 scale,
        float64 normalized, float64 rho, float64 learning_rate,
        float64 epsilon
        """,
        """
        float64 updated, float64 new_scale, float64 new_normalized,
        raw uint32 invalid
        """,
        """
        new_scale = fmax(scale, fabs(gradient));
        double safe_scale = new_scale == 0.0 ? 1.0 : new_scale;
        double previous_ratio = scale / safe_scale;
        double gradient_ratio = fabs(gradient) / safe_scale;
        new_normalized = (
            rho * normalized * previous_ratio * previous_ratio
            + (1.0 - rho) * gradient_ratio * gradient_ratio
        );
        if (new_scale == 0.0) new_normalized = 0.0;
        double root_moment = new_scale * sqrt(new_normalized);
        updated = parameter - (
            learning_rate * gradient / (root_moment + epsilon)
        );
        if (
            !isfinite(parameter)
            || !isfinite(gradient)
            || !isfinite(scale)
            || !isfinite(normalized)
            || !isfinite(updated)
            || !isfinite(new_scale)
            || !isfinite(new_normalized)
        ) atomicExch(&invalid[0], 1U);
        """,
        "tensors_rmsprop_batch",
    )

def _optimizer_scalar_batch(
    values: Sequence[float],
    references: Sequence[Tensor],
    numpy: Any,
) -> Any:
    """Expand one scalar per parameter into its batched element layout."""
    if values and all(value == values[0] for value in values[1:]):
        return float(values[0])
    scalars = numpy.asarray(tuple(values), dtype=numpy.float64)
    counts = numpy.asarray(
        tuple(reference.size for reference in references),
        dtype=numpy.int64,
    )
    return numpy.repeat(scalars, counts)

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
        parameters,
        gradients,
        scales,
        scaled_values,
    )
    if partitions is None:
        return None
    if len(partitions) > 1:
        merged: list[list[Storage | None]] = [
            [None] * len(parameters) for _ in range(3)
        ]
        for indices in partitions:
            result = rmsprop_updates(
                *(
                    _optimizer_partition(group, indices)
                    for group in groups
                ),
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
    parameter_values, gradient_values, scale_values, normalized_values = arrays
    if _array_kind(numpy) == "cuda":
        parameter_result = numpy.empty_like(parameter_values)
        new_scales = numpy.empty_like(scale_values)
        new_scaled = numpy.empty_like(normalized_values)
        invalid = _optimizer_invalid_flag(numpy)
        _cuda_rmsprop_batch_kernel()(
            parameter_values,
            gradient_values,
            scale_values,
            normalized_values,
            rho,
            learning_rate,
            epsilon,
            parameter_result,
            new_scales,
            new_scaled,
            invalid,
        )
        if bool(invalid[0]):
            return None
    else:
        with _errstate(
            numpy,
            divide="ignore",
            over="ignore",
            under="ignore",
            invalid="ignore",
        ):
            new_scales = numpy.maximum(
                scale_values,
                numpy.abs(gradient_values),
            )
            safe_scales = numpy.where(new_scales == 0.0, 1.0, new_scales)
            previous_ratio = scale_values / safe_scales
            gradient_ratio = numpy.abs(gradient_values) / safe_scales
            new_scaled = (
                rho * normalized_values * previous_ratio * previous_ratio
                + (1.0 - rho) * gradient_ratio * gradient_ratio
            )
            new_scaled = numpy.where(new_scales == 0.0, 0.0, new_scaled)
            root_moment = new_scales * numpy.sqrt(new_scaled)
            parameter_result = parameter_values - (
                learning_rate * gradient_values / (root_moment + epsilon)
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
        _split_optimizer_storage(parameter_result, parameters, numpy),
        _split_optimizer_storage(new_scales, gradients, numpy),
        _split_optimizer_storage(new_scaled, gradients, numpy),
    )
    if any(result is None for result in results):
        return None
    return cast("tuple[tuple[Storage, ...], ...]", results)

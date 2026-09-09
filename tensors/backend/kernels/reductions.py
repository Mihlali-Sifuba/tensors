"""Reduction kernels and the stable-summation machinery behind them.

Also home to the shifted-exponential terms shared by ``logsumexp`` and the
softmax family in :mod:`~tensors.backend.kernels.nn`."""

from __future__ import annotations

import math
from typing import Any, TYPE_CHECKING

from ..storage import Storage
from .core import _errstate, _finite_operands, _numpy, _storage, _view

if TYPE_CHECKING:
    from ...dtype import DataType
    from ...tensor import Tensor
    from ..types import (
        ArgExtremumOperation,
        DifferentiableReductionOperation,
        ReductionOperation,
    )

def _normalization_terms(
    values: Any,
    axis: int | tuple[int, ...],
    numpy: Any,
) -> tuple[Any, Any, Any]:
    """Return stable maxima, corrections, and probabilities."""
    maximum = numpy.max(values, axis=axis, keepdims=True)
    with _errstate(numpy, over="ignore", under="ignore", invalid="ignore"):
        deltas = values - maximum
        maxima = numpy.sum(deltas == 0.0, axis=axis, keepdims=True)
        tails = numpy.sum(
            numpy.where(deltas == 0.0, 0.0, numpy.exp(deltas)),
            axis=axis,
            keepdims=True,
        )
        correction = numpy.log(maxima) + numpy.log1p(tails / maxima)
        probabilities = numpy.exp(deltas - correction)
    return maximum, correction, probabilities

def logsumexp(
    value: Tensor,
    axes: tuple[int, ...],
    *,
    keepdims: bool,
    dtype: DataType,
    output_shape: tuple[int, ...],
) -> Storage | None:
    """Run a stable log-sum-exp reduction on finite values."""
    numpy = _numpy()
    values = _view(value, numpy).astype(numpy.float64, copy=False)
    maximum, correction, probabilities = _normalization_terms(
        values,
        axes,
        numpy,
    )
    with _errstate(numpy, over="ignore", invalid="ignore"):
        result = maximum + correction
    if not keepdims and axes:
        result = numpy.squeeze(result, axis=axes)
    if not _finite_operands(values, probabilities, result, numpy=numpy):
        return None
    return _storage(
        result,
        dtype=dtype,
        output_shape=output_shape,
        numpy=numpy,
    )

def logsumexp_gradient(
    grad: Tensor,
    value: Tensor,
    axes: tuple[int, ...],
    *,
    keepdims: bool,
) -> Storage | None:
    """Run a stable log-sum-exp VJP on finite values."""
    numpy = _numpy()
    values = _view(value, numpy).astype(numpy.float64, copy=False)
    upstream = _view(grad, numpy).astype(numpy.float64, copy=False)
    _, _, probabilities = _normalization_terms(values, axes, numpy)
    expanded_shape = tuple(
        1 if dimension in axes else size
        for dimension, size in enumerate(value.shape)
    )
    try:
        expanded = upstream.reshape(expanded_shape)
    except ValueError:
        return None
    with _errstate(numpy, over="ignore", under="ignore", invalid="ignore"):
        result = expanded * probabilities
    if not _finite_operands(
        values,
        upstream,
        probabilities,
        result,
        numpy=numpy,
    ):
        return None
    return _storage(
        result,
        dtype=grad.dtype,
        output_shape=value.shape,
        numpy=numpy,
    )

def _summation_guard(
    values: Any,
    numpy: Any,
    *,
    axes: tuple[int, ...] | None = None,
    keepdims: bool = False,
    mixed_signs: bool = True,
) -> Any:
    """Build one provider-native safety decision for ordinary summation."""
    if values.size == 0:
        return numpy.asarray(True)
    minimum = numpy.min(values, axis=axes, keepdims=keepdims)
    maximum = numpy.max(values, axis=axes, keepdims=keepdims)
    subnormal = numpy.any(
        (values != 0.0)
        & (numpy.abs(values) < numpy.finfo(numpy.float64).tiny),
        axis=axes,
        keepdims=keepdims,
    )
    valid = (
        numpy.isfinite(minimum)
        & numpy.isfinite(maximum)
        & ~subnormal
    )
    if mixed_signs:
        absolute = numpy.abs(values)
        largest = numpy.max(absolute, axis=axes, keepdims=keepdims)
        smallest = numpy.min(
            numpy.where(absolute == 0.0, numpy.inf, absolute),
            axis=axes,
            keepdims=keepdims,
        )
        comparable_magnitudes = (largest == 0.0) | (
            smallest >= largest * numpy.finfo(numpy.float64).eps
        )
        mixed = (minimum < 0.0) & (maximum > 0.0)
        valid &= ~(mixed & ~comparable_magnitudes)
    return numpy.all(valid)

def _scaled_sum(values: Any, axes: tuple[int, ...], numpy: Any) -> Any:
    """Sum comparable finite values without overflowing intermediates."""
    if not axes:
        return values
    scale = numpy.max(numpy.abs(values), axis=axes, keepdims=True)
    safe_scale = numpy.where(scale == 0.0, 1.0, scale)
    with _errstate(numpy, over="ignore", under="ignore", invalid="ignore"):
        normalized = values / safe_scale
        result = numpy.sum(normalized, axis=axes, keepdims=True) * scale
    return numpy.where(scale == 0.0, 0.0, result)

def _scaled_product_sum(
    left: Any,
    right: Any,
    axes: tuple[int, ...],
    numpy: Any,
) -> Any:
    """Evaluate a product reduction through normalized device operands."""
    reduction_axes: tuple[int, ...] | None = axes if axes else None
    left_scale = numpy.max(numpy.abs(left), axis=reduction_axes, keepdims=True)
    right_scale = numpy.max(numpy.abs(right), axis=reduction_axes, keepdims=True)
    safe_left = numpy.where(left_scale == 0.0, 1.0, left_scale)
    safe_right = numpy.where(right_scale == 0.0, 1.0, right_scale)
    with _errstate(numpy, over="ignore", under="ignore", invalid="ignore"):
        normalized_terms = (left / safe_left) * (right / safe_right)
        normalized_sum = (
            numpy.sum(normalized_terms, axis=axes, keepdims=True)
            if axes
            else normalized_terms
        )
        log_magnitude = (
            numpy.log(numpy.abs(normalized_sum))
            + numpy.log(safe_left)
            + numpy.log(safe_right)
        )
        restored = numpy.copysign(numpy.exp(log_magnitude), normalized_sum)
    zero = (
        (left_scale == 0.0)
        | (right_scale == 0.0)
        | (normalized_sum == 0.0)
    )
    return numpy.where(zero, 0.0, restored)

def reduction(
    operation: ReductionOperation,
    value: Tensor,
    axes: tuple[int, ...],
    *,
    keepdims: bool,
    dtype: DataType,
    output_shape: tuple[int, ...],
) -> Storage | None:
    """Run a numerically guarded NumPy reduction."""
    if value.size == 0:
        return None

    numpy = _numpy()
    axis = axes
    if operation in {"min", "max"}:
        values = _view(value, numpy)
        function = numpy.min if operation == "min" else numpy.max
        result = function(values, axis=axis, keepdims=keepdims)
    elif operation == "prod":
        values = _view(value, numpy)
        working = (
            values.astype(object)
            if dtype.kind == "integer"
            else values.astype(numpy.float64, copy=False)
        )
        with _errstate(numpy, over="ignore", under="ignore", invalid="ignore"):
            result = numpy.prod(working, axis=axis, keepdims=keepdims)
    else:
        values = _view(value, numpy).astype(numpy.float64, copy=False)

    if operation in {"sum", "mean"}:
        if operation == "sum" and dtype.kind == "integer":
            return None
        with _errstate(numpy, over="ignore", under="ignore", invalid="ignore"):
            direct = numpy.sum(values, axis=axis, keepdims=True)
            if operation == "mean":
                count = math.prod(value.shape[item] for item in axis)
                direct = direct / count

        # A finite same-sign reduction cannot lose values through
        # cancellation, and a finite direct result proves that no partial
        # overflow escaped the provider.  This common NumPy case only needs
        # min/max plus the sum; mixed-sign or extreme data retains the full
        # scaled guard below.
        from ..config import get_backend

        ordinary = False
        if get_backend() == "numpy":
            minimum = numpy.min(values, axis=axis, keepdims=True)
            maximum = numpy.max(values, axis=axis, keepdims=True)
            ordinary = bool(numpy.all(
                numpy.isfinite(minimum)
                & numpy.isfinite(maximum)
                & numpy.isfinite(direct)
                & ((minimum >= 0.0) | (maximum <= 0.0))
            ))
        if ordinary:
            result = direct
        else:
            with _errstate(
                numpy,
                over="ignore",
                under="ignore",
                invalid="ignore",
            ):
                scaled = _scaled_sum(values, axis, numpy)
                if operation == "mean":
                    scaled = scaled / count
            safe = _summation_guard(
                values,
                numpy,
                axes=axis,
                keepdims=True,
            )
            direct_safe = safe & numpy.all(numpy.isfinite(direct))
            result = numpy.where(direct_safe, direct, scaled)
            valid = safe & ~numpy.any(numpy.isnan(result))
            if not bool(valid):
                return None
        if not keepdims and axis:
            result = numpy.squeeze(result, axis=axis)
    elif operation in {"variance", "std"}:
        with _errstate(numpy, over="ignore", under="ignore", invalid="ignore"):
            center = numpy.mean(values, axis=axis, keepdims=True)
            centered = values - center
            scale = numpy.max(
                numpy.abs(centered),
                axis=axis,
                keepdims=True,
            )
            safe_scale = numpy.where(scale == 0.0, 1.0, scale)
            normalized = centered / safe_scale
            normalized_variance = numpy.mean(
                normalized * normalized,
                axis=axis,
                keepdims=keepdims,
            )
            output_scale = (
                scale if keepdims else numpy.squeeze(scale, axis=axis)
            )
            deviation = output_scale * numpy.sqrt(normalized_variance)
            result = (
                deviation * deviation
                if operation == "variance"
                else deviation
            )
        valid = (
            numpy.all(numpy.isfinite(values))
            & numpy.all(numpy.isfinite(centered))
            & numpy.all(numpy.isfinite(result))
        )
        if not bool(valid):
            return None
    elif operation == "norm":
        with _errstate(numpy, over="ignore", under="ignore", invalid="ignore"):
            absolute = numpy.abs(values)
            scale = numpy.max(absolute, axis=axis, keepdims=True)
            safe_scale = numpy.where(scale == 0.0, 1.0, scale)
            normalized = values / safe_scale
            normalized_magnitude = numpy.sqrt(
                numpy.sum(
                    normalized * normalized,
                    axis=axis,
                    keepdims=keepdims,
                )
            )
            output_scale = (
                scale if keepdims else numpy.squeeze(scale, axis=axis)
            )
            result = output_scale * normalized_magnitude
        valid = numpy.all(numpy.isfinite(values)) & numpy.all(
            numpy.isfinite(result)
        )
        if not bool(valid):
            return None

    return _storage(
        result,
        dtype=dtype,
        output_shape=output_shape,
        numpy=numpy,
    )

def reduction_gradient(
    operation: DifferentiableReductionOperation,
    grad: Tensor,
    value: Tensor,
    axes: tuple[int, ...],
    *,
    keepdims: bool,
) -> Storage | None:
    """Run fused VJPs for reductions with regular native fast paths."""
    numpy = _numpy()
    values = _view(value, numpy).astype(numpy.float64, copy=False)
    upstream = _view(grad, numpy).astype(numpy.float64, copy=False)
    expanded_shape = tuple(
        1 if dimension in axes else size
        for dimension, size in enumerate(value.shape)
    )
    try:
        expanded = upstream.reshape(expanded_shape)
    except ValueError:
        return None
    count = 1
    for axis in axes:
        count *= value.shape[axis]

    with _errstate(

        numpy,
        divide="ignore",
        over="ignore",
        under="ignore",
        invalid="ignore",
    ):
        if operation == "sum":
            # `broadcast_to` returns a view of the upstream gradient.
            result = numpy.broadcast_to(expanded, value.shape).copy()
        elif operation == "mean":
            if count == 0:
                return None
            result = numpy.broadcast_to(expanded / count, value.shape)
        elif operation == "variance":
            if count == 0:
                return None
            center = numpy.mean(values, axis=axes, keepdims=True)
            centered = values - center
            scale = numpy.max(
                numpy.abs(centered),
                axis=axes,
                keepdims=True,
            )
            safe_scale = numpy.where(scale == 0.0, 1.0, scale)
            normalized = centered / safe_scale
            result = expanded * normalized * scale * (2.0 / count)
            valid = (
                numpy.all(numpy.isfinite(values))
                & numpy.all(numpy.isfinite(centered))
                & numpy.all(numpy.isfinite(result))
            )
            if not bool(valid):
                return None
        elif operation == "std":
            if not bool(numpy.all(numpy.isfinite(values))) or count == 0:
                return None
            scale = numpy.max(numpy.abs(values), axis=axes, keepdims=True)
            safe_scale = numpy.where(scale == 0.0, 1.0, scale)
            normalized = values / safe_scale
            center = numpy.mean(normalized, axis=axes, keepdims=True)
            centered = normalized - center
            deviation = numpy.sqrt(
                numpy.mean(centered * centered, axis=axes, keepdims=True)
            )
            derivative = numpy.where(
                deviation == 0.0,
                0.0,
                centered / (count * deviation),
            )
            result = expanded * derivative
        elif operation == "prod":
            if not bool(numpy.all(numpy.isfinite(values))):
                return None
            zero_count = numpy.sum(values == 0.0, axis=axes, keepdims=True)
            product = numpy.prod(values, axis=axes, keepdims=True)
            nonzero_product = numpy.prod(
                numpy.where(values == 0.0, 1.0, values),
                axis=axes,
                keepdims=True,
            )
            unsafe = (
                ((zero_count == 0) & ((product == 0.0) | ~numpy.isfinite(product)))
                | (
                    (zero_count == 1)
                    & (
                        (nonzero_product == 0.0)
                        | ~numpy.isfinite(nonzero_product)
                    )
                )
            )
            if bool(numpy.any(unsafe)):
                return None
            derivative = numpy.where(
                zero_count == 0,
                product / values,
                numpy.where(
                    (zero_count == 1) & (values == 0.0),
                    nonzero_product,
                    0.0,
                ),
            )
            result = expanded * derivative
        else:
            has_nan = numpy.any(numpy.isnan(values), axis=axes, keepdims=True)
            function = numpy.min if operation == "min" else numpy.max
            extreme = function(values, axis=axes, keepdims=True)
            selected = values == extreme
            ties = numpy.sum(selected, axis=axes, keepdims=True)
            result = numpy.where(
                has_nan,
                numpy.nan,
                expanded * selected / ties,
            )
    return _storage(
        result,
        dtype=grad.dtype,
        output_shape=value.shape,
        numpy=numpy,
    )

def arg_extremum(
    operation: ArgExtremumOperation,
    value: Tensor,
    axis: int | None,
    *,
    keepdims: bool,
    output_shape: tuple[int, ...],
) -> Storage | None:
    """Run a first-occurrence argmin or argmax reduction."""
    if value.size == 0:
        return None
    from ...dtype import int64

    numpy = _numpy()
    values = _view(value, numpy)
    function = numpy.argmin if operation == "argmin" else numpy.argmax
    result = function(values, axis=axis, keepdims=keepdims)
    return _storage(
        result,
        dtype=int64,
        output_shape=output_shape,
        numpy=numpy,
    )

def _sum_axes(
    source_shape: tuple[int, ...],
    target_shape: tuple[int, ...],
) -> tuple[tuple[int, ...], tuple[int, ...]] | None:
    """Return padded target shape and axes reduced after broadcasting."""
    if len(target_shape) > len(source_shape):
        return None
    padded = (1,) * (len(source_shape) - len(target_shape)) + target_shape
    axes = []
    for axis, (source, target) in enumerate(zip(source_shape, padded)):
        if source == target:
            continue
        if target != 1:
            return None
        axes.append(axis)
    return padded, tuple(axes)

def _stable_sum_candidate(values: Any, axes: tuple[int, ...], numpy: Any) -> Any:
    """Return a provider-native guard for ordinary reduction summation."""
    if axes:
        return _summation_guard(
            values,
            numpy,
            axes=axes,
            keepdims=True,
        )
    return _summation_guard(values, numpy, mixed_signs=False)

def sum_to_shape(
    gradient: Tensor,
    shape: tuple[int, ...],
) -> Storage | None:
    """Reduce a broadcast gradient using guarded native summation."""
    if gradient.dtype.kind == "integer":
        return None
    layout = _sum_axes(gradient.shape, shape)
    if layout is None:
        return None
    _, axes = layout
    numpy = _numpy()
    values = _view(gradient, numpy).astype(numpy.float64, copy=False)
    safe = _stable_sum_candidate(values, axes, numpy)
    result = _scaled_sum(values, axes, numpy)
    valid = safe & ~numpy.any(numpy.isnan(result))
    if not bool(valid):
        return None
    return _storage(
        numpy.asarray(result).reshape(shape),
        dtype=gradient.dtype,
        output_shape=shape,
        numpy=numpy,
    )

def sum_products_to_shape(
    gradient: Tensor,
    factor: Tensor,
    shape: tuple[int, ...],
) -> Storage | None:
    """Multiply and reduce broadcast VJP terms in one guarded kernel."""
    if gradient.dtype.kind == "integer":
        return None
    numpy = _numpy()
    try:
        left, right = numpy.broadcast_arrays(
            _view(gradient, numpy).astype(numpy.float64, copy=False),
            _view(factor, numpy).astype(numpy.float64, copy=False),
        )
    except ValueError:
        return None
    layout = _sum_axes(tuple(left.shape), shape)
    if layout is None:
        return None
    _, axes = layout
    finite = numpy.all(numpy.isfinite(left)) & numpy.all(numpy.isfinite(right))
    nonzero = (left != 0.0) & (right != 0.0)
    reduction_axes: tuple[int, ...] | None = axes if axes else None
    with _errstate(numpy, divide="ignore", invalid="ignore"):
        log_terms = numpy.log(numpy.abs(left)) + numpy.log(numpy.abs(right))
    largest_log = numpy.max(
        numpy.where(nonzero, log_terms, -numpy.inf),
        axis=reduction_axes,
        keepdims=True,
    )
    smallest_log = numpy.min(
        numpy.where(nonzero, log_terms, numpy.inf),
        axis=reduction_axes,
        keepdims=True,
    )
    any_nonzero = numpy.any(nonzero, axis=reduction_axes, keepdims=True)
    comparable = (~any_nonzero) | (
        largest_log - smallest_log
        <= -math.log(numpy.finfo(numpy.float64).eps)
    )
    with _errstate(numpy, over="ignore", under="ignore", invalid="ignore"):
        products = left * right
        direct = (
            numpy.sum(products, axis=axes, keepdims=True)
            if axes
            else products
        )
    lost_range = (~numpy.isfinite(products)) | (
        (products == 0.0) & nonzero
    )
    direct_safe = (
        ~numpy.any(lost_range)
        & _stable_sum_candidate(products, axes, numpy)
        & ~numpy.any(numpy.isnan(direct))
    )
    stable = _scaled_product_sum(left, right, axes, numpy)
    result = numpy.where(direct_safe, direct, stable)
    valid = (
        finite
        & (direct_safe | numpy.all(comparable))
        & ~numpy.any(numpy.isnan(result))
    )
    if not bool(valid):
        return None
    return _storage(
        numpy.asarray(result).reshape(shape),
        dtype=gradient.dtype,
        output_shape=shape,
        numpy=numpy,
    )

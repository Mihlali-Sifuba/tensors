"""General reductions and their vector-Jacobian products."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from ...storage import Storage
from ..core import _errstate, _numpy, _storage, _view
from .stability import _scaled_sum, _summation_guard

if TYPE_CHECKING:
    from ....dtype import DataType
    from ....tensor import Tensor
    from ...types import (
        DifferentiableReductionOperation,
        ReductionOperation,
    )

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
        from ...config import get_backend

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

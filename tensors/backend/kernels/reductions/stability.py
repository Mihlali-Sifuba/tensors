"""Summation guards and scaled accumulation.

The primitives here keep reductions finite when naive accumulation would
overflow or lose every small magnitude."""

from __future__ import annotations

from typing import Any

from ..core import _errstate

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

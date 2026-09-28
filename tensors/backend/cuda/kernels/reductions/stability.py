"""Summation guards and scaled accumulation.

The primitives here keep reductions finite when naive accumulation would
overflow or lose every small magnitude.
"""

from __future__ import annotations
import cupy
from typing import Any
from tensors.backend.cuda.conversion import _errstate


def _summation_guard(
    values: Any,
    *,
    axes: tuple[int, ...] | None = None,
    keepdims: bool = False,
    mixed_signs: bool = True,
) -> Any:
    """Build one provider-native safety decision for ordinary summation."""
    if values.size == 0:
        return cupy.asarray(True)
    minimum = cupy.min(values, axis=axes, keepdims=keepdims)
    maximum = cupy.max(values, axis=axes, keepdims=keepdims)
    subnormal = cupy.any(
        (values != 0.0) & (cupy.abs(values) < cupy.finfo(cupy.float64).tiny),
        axis=axes,
        keepdims=keepdims,
    )
    valid = cupy.isfinite(minimum) & cupy.isfinite(maximum) & ~subnormal
    if mixed_signs:
        absolute = cupy.abs(values)
        largest = cupy.max(absolute, axis=axes, keepdims=keepdims)
        smallest = cupy.min(
            cupy.where(absolute == 0.0, cupy.inf, absolute),
            axis=axes,
            keepdims=keepdims,
        )
        comparable_magnitudes = (largest == 0.0) | (
            smallest >= largest * cupy.finfo(cupy.float64).eps
        )
        mixed = (minimum < 0.0) & (maximum > 0.0)
        valid &= ~(mixed & ~comparable_magnitudes)
    return cupy.all(valid)


def _scaled_sum(values: Any, axes: tuple[int, ...]) -> Any:
    """Sum comparable finite values without overflowing intermediates."""
    if not axes:
        return values
    scale = cupy.max(cupy.abs(values), axis=axes, keepdims=True)
    safe_scale = cupy.where(scale == 0.0, 1.0, scale)
    with _errstate(over="ignore", under="ignore", invalid="ignore"):
        normalized = values / safe_scale
        result = cupy.sum(normalized, axis=axes, keepdims=True) * scale
    return cupy.where(scale == 0.0, 0.0, result)


def _scaled_product_sum(left: Any, right: Any, axes: tuple[int, ...]) -> Any:
    """Evaluate a product reduction through normalized device operands."""
    reduction_axes: tuple[int, ...] | None = axes if axes else None
    left_scale = cupy.max(cupy.abs(left), axis=reduction_axes, keepdims=True)
    right_scale = cupy.max(cupy.abs(right), axis=reduction_axes, keepdims=True)
    safe_left = cupy.where(left_scale == 0.0, 1.0, left_scale)
    safe_right = cupy.where(right_scale == 0.0, 1.0, right_scale)
    with _errstate(over="ignore", under="ignore", invalid="ignore"):
        normalized_terms = left / safe_left * (right / safe_right)
        normalized_sum = (
            cupy.sum(normalized_terms, axis=axes, keepdims=True)
            if axes
            else normalized_terms
        )
        log_magnitude = (
            cupy.log(cupy.abs(normalized_sum))
            + cupy.log(safe_left)
            + cupy.log(safe_right)
        )
        restored = cupy.copysign(cupy.exp(log_magnitude), normalized_sum)
    zero = (left_scale == 0.0) | (right_scale == 0.0) | (normalized_sum == 0.0)
    return cupy.where(zero, 0.0, restored)


def _sum_axes(
    source_shape: tuple[int, ...], target_shape: tuple[int, ...]
) -> tuple[int, ...] | None:
    """Return the axes reduced after broadcasting, or ``None`` to decline.

    The derivation is :meth:`~tensors.shape.Shape.stretched_axes_from`, which
    states it once for the whole package. A kernel answers a shape it cannot
    reduce by declining rather than raising, so the shape error becomes a
    ``None`` here and the dispatcher reports it.
    """
    from tensors.shape import Shape

    try:
        return Shape.from_iterable(source_shape).stretched_axes_from(target_shape)
    except ValueError:
        return None


def _stable_sum_candidate(values: Any, axes: tuple[int, ...]) -> Any:
    """Return a provider-native guard for ordinary reduction summation."""
    if axes:
        return _summation_guard(values, axes=axes, keepdims=True)
    return _summation_guard(values, mixed_signs=False)

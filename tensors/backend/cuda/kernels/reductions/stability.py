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

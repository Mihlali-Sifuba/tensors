"""Matrix products and their vector-Jacobian products."""

from __future__ import annotations
import numpy
from typing import Any
from tensors.backend.numpy.conversion import _errstate
from tensors.backend.numpy.kernels.reductions.stability import _scaled_sum
from tensors.backend.numpy.kernels.reductions.stability import _stable_sum_candidate
from tensors.backend.numpy.kernels.reductions.stability import _sum_axes


def _comparable_finite_values(values: Any) -> bool:
    """Return whether global scaling can retain every nonzero magnitude."""
    if not bool(numpy.all(numpy.isfinite(values))):
        return False
    absolute = numpy.abs(values)
    largest = numpy.max(absolute)
    smallest = numpy.min(numpy.where(absolute == 0.0, numpy.inf, absolute))
    return bool(
        (largest == 0.0) | (smallest >= largest * numpy.finfo(numpy.float64).eps)
    )


def _scaled_matmul(left: Any, right: Any) -> Any:
    """Calculate a matrix product without overflowing temporary products."""
    left_scale = numpy.max(numpy.abs(left))
    right_scale = numpy.max(numpy.abs(right))
    safe_left = numpy.where(left_scale == 0.0, 1.0, left_scale)
    safe_right = numpy.where(right_scale == 0.0, 1.0, right_scale)
    with _errstate(over="ignore", under="ignore", invalid="ignore"):
        normalized = numpy.matmul(left / safe_left, right / safe_right)
        log_magnitude = (
            numpy.log(numpy.abs(normalized))
            + numpy.log(safe_left)
            + numpy.log(safe_right)
        )
        restored = numpy.copysign(numpy.exp(log_magnitude), normalized)
    zero = (left_scale == 0.0) | (right_scale == 0.0) | (normalized == 0.0)
    return numpy.where(zero, 0.0, restored)


def _matrix_view(value: Any, vector: bool, *, left: bool) -> Any:
    """Promote a vector to the matrix shape used by matmul differentiation."""
    if not vector:
        return value
    if left:
        return value.reshape((1, value.shape[0]))
    return value.reshape((value.shape[0], 1))


def _matrix_gradient_view(gradient: Any, left_vector: bool, right_vector: bool) -> Any:
    """Restore the two matrix axes omitted from a public matmul result."""
    if left_vector and right_vector:
        return gradient.reshape((1, 1))
    if left_vector:
        return gradient.reshape(gradient.shape[:-1] + (1, gradient.shape[-1]))
    if right_vector:
        return gradient.reshape(gradient.shape + (1,))
    return gradient


def _reduce_matrix_gradient(values: Any, shape: tuple[int, ...]) -> Any | None:
    """Reduce broadcast batch axes back to one operand's matrix shape."""
    axes = _sum_axes(tuple(values.shape), shape)
    if axes is None:
        return None
    safe = _stable_sum_candidate(values, axes)
    if not bool(safe):
        return None
    if axes:
        with _errstate(over="ignore", under="ignore", invalid="ignore"):
            direct = numpy.sum(values, axis=axes, keepdims=True)
        scaled = _scaled_sum(values, axes)
        values = numpy.where(numpy.isfinite(direct), direct, scaled)
        if bool(numpy.any(numpy.isnan(values))):
            return None
    return values.reshape(shape)

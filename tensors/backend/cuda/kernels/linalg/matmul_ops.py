"""Matrix products and their vector-Jacobian products."""

from __future__ import annotations
import cupy
from typing import Any
from tensors.backend.cuda.conversion import _errstate
from tensors.backend.cuda.kernels.reductions.stability import _scaled_sum
from tensors.backend.cuda.kernels.reductions.stability import _stable_sum_candidate
from tensors.backend.cuda.kernels.reductions.stability import _sum_axes


def _comparable_finite_values(values: Any) -> bool:
    """Return whether global scaling can retain every nonzero magnitude."""
    if not bool(cupy.all(cupy.isfinite(values))):
        return False
    absolute = cupy.abs(values)
    largest = cupy.max(absolute)
    smallest = cupy.min(cupy.where(absolute == 0.0, cupy.inf, absolute))
    return bool((largest == 0.0) | (smallest >= largest * cupy.finfo(cupy.float64).eps))


def _scaled_matmul(left: Any, right: Any) -> Any:
    """Calculate a matrix product without overflowing temporary products."""
    left_scale = cupy.max(cupy.abs(left))
    right_scale = cupy.max(cupy.abs(right))
    safe_left = cupy.where(left_scale == 0.0, 1.0, left_scale)
    safe_right = cupy.where(right_scale == 0.0, 1.0, right_scale)
    with _errstate(over="ignore", under="ignore", invalid="ignore"):
        normalized = cupy.matmul(left / safe_left, right / safe_right)
        log_magnitude = (
            cupy.log(cupy.abs(normalized)) + cupy.log(safe_left) + cupy.log(safe_right)
        )
        restored = cupy.copysign(cupy.exp(log_magnitude), normalized)
    zero = (left_scale == 0.0) | (right_scale == 0.0) | (normalized == 0.0)
    return cupy.where(zero, 0.0, restored)


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
    layout = _sum_axes(tuple(values.shape), shape)
    if layout is None:
        return None
    _, axes = layout
    safe = _stable_sum_candidate(values, axes)
    if not bool(safe):
        return None
    if axes:
        with _errstate(over="ignore", under="ignore", invalid="ignore"):
            direct = cupy.sum(values, axis=axes, keepdims=True)
        scaled = _scaled_sum(values, axes)
        values = cupy.where(cupy.isfinite(direct), direct, scaled)
        if bool(cupy.any(cupy.isnan(values))):
            return None
    return values.reshape(shape)

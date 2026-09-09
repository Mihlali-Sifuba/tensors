"""Matrix products and their vector-Jacobian products."""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

from ...storage import Storage
from ..core import _errstate, _finite_operands, _numpy, _storage, _view
from ..reductions.stability import (
    _scaled_sum,
    _stable_sum_candidate,
    _sum_axes,
)

if TYPE_CHECKING:
    from ....dtype import DataType
    from ....tensor import Tensor

def _comparable_finite_values(values: Any, numpy: Any) -> bool:
    """Return whether global scaling can retain every nonzero magnitude."""
    if not bool(numpy.all(numpy.isfinite(values))):
        return False
    absolute = numpy.abs(values)
    largest = numpy.max(absolute)
    smallest = numpy.min(numpy.where(absolute == 0.0, numpy.inf, absolute))
    return bool(
        (largest == 0.0)
        | (smallest >= largest * numpy.finfo(numpy.float64).eps)
    )

def _scaled_matmul(left: Any, right: Any, numpy: Any) -> Any:
    """Calculate a matrix product without overflowing temporary products."""
    left_scale = numpy.max(numpy.abs(left))
    right_scale = numpy.max(numpy.abs(right))
    safe_left = numpy.where(left_scale == 0.0, 1.0, left_scale)
    safe_right = numpy.where(right_scale == 0.0, 1.0, right_scale)
    with _errstate(numpy, over="ignore", under="ignore", invalid="ignore"):
        normalized = numpy.matmul(left / safe_left, right / safe_right)
        log_magnitude = (
            numpy.log(numpy.abs(normalized))
            + numpy.log(safe_left)
            + numpy.log(safe_right)
        )
        restored = numpy.copysign(numpy.exp(log_magnitude), normalized)
    zero = (
        (left_scale == 0.0)
        | (right_scale == 0.0)
        | (normalized == 0.0)
    )
    return numpy.where(zero, 0.0, restored)

def matmul(
    left: Tensor,
    right: Tensor,
    *,
    dtype: DataType,
    output_shape: tuple[int, ...],
) -> Storage | None:
    """Return a NumPy matrix product or defer to the reference implementation."""
    if dtype.kind != "floating":
        return None

    numpy = _numpy()

    # The reference implementation accumulates float32 products in Python's
    # double precision and casts only the final result. Match that behavior by
    # using float64 as the NumPy working dtype for every supported float result.
    try:
        left_array = _view(left, numpy).astype(numpy.float64, copy=False)
        right_array = _view(right, numpy).astype(numpy.float64, copy=False)
    except ValueError:
        return None
    with _errstate(numpy, over="ignore", under="ignore", invalid="ignore"):
        result = numpy.matmul(left_array, right_array)

    # Recover device-side when comparable finite operands overflow temporary
    # products. Highly disparate magnitudes still use the exact reference path.
    if not bool(numpy.all(numpy.isfinite(result))):
        if not (
            _comparable_finite_values(left_array, numpy)
            and _comparable_finite_values(right_array, numpy)
        ):
            return None
        result = _scaled_matmul(left_array, right_array, numpy)
        if bool(numpy.any(numpy.isnan(result))):
            return None
    return _storage(
        result,
        dtype=dtype,
        output_shape=output_shape,
        numpy=numpy,
    )

def _matrix_view(value: Any, vector: bool, *, left: bool) -> Any:
    """Promote a vector to the matrix shape used by matmul differentiation."""
    if not vector:
        return value
    if left:
        return value.reshape((1, value.shape[0]))
    return value.reshape((value.shape[0], 1))

def _matrix_gradient_view(
    gradient: Any,
    left_vector: bool,
    right_vector: bool,
) -> Any:
    """Restore the two matrix axes omitted from a public matmul result."""
    if left_vector and right_vector:
        return gradient.reshape((1, 1))
    if left_vector:
        return gradient.reshape(gradient.shape[:-1] + (1, gradient.shape[-1]))
    if right_vector:
        return gradient.reshape(gradient.shape + (1,))
    return gradient

def _reduce_matrix_gradient(
    values: Any,
    shape: tuple[int, ...],
    numpy: Any,
) -> Any | None:
    """Reduce broadcast batch axes back to one operand's matrix shape."""
    layout = _sum_axes(tuple(values.shape), shape)
    if layout is None:
        return None
    _, axes = layout
    safe = _stable_sum_candidate(values, axes, numpy)
    if not bool(safe):
        return None
    if axes:
        with _errstate(numpy, over="ignore", under="ignore", invalid="ignore"):
            direct = numpy.sum(values, axis=axes, keepdims=True)
        scaled = _scaled_sum(values, axes, numpy)
        values = numpy.where(numpy.isfinite(direct), direct, scaled)
        if bool(numpy.any(numpy.isnan(values))):
            return None
    return values.reshape(shape)

def matmul_gradient(
    grad: Tensor,
    left: Tensor,
    right: Tensor,
    *,
    needs_input_grad: tuple[bool, ...] = (True, True),
) -> tuple[Storage | None, Storage | None] | None:
    """Run the requested native vector-Jacobian products for matmul."""
    if grad.dtype.kind != "floating":
        return None
    numpy = _numpy()
    left_vector = left.ndim == 1
    right_vector = right.ndim == 1
    try:
        upstream = _view(grad, numpy).astype(numpy.float64, copy=False)
        left_values = _view(left, numpy).astype(numpy.float64, copy=False)
        right_values = _view(right, numpy).astype(numpy.float64, copy=False)
    except ValueError:
        return None
    if not _finite_operands(
        upstream,
        left_values,
        right_values,
        numpy=numpy,
    ):
        return None

    left_matrix = _matrix_view(
        left_values,
        left_vector,
        left=True,
    )
    right_matrix = _matrix_view(
        right_values,
        right_vector,
        left=False,
    )
    matrix_grad = _matrix_gradient_view(
        upstream,
        left_vector,
        right_vector,
    )
    need_left, need_right = needs_input_grad
    with _errstate(numpy, over="ignore", under="ignore", invalid="ignore"):
        left_result = (
            numpy.matmul(matrix_grad, numpy.swapaxes(right_matrix, -1, -2))
            if need_left
            else None
        )
        right_result = (
            numpy.matmul(numpy.swapaxes(left_matrix, -1, -2), matrix_grad)
            if need_right
            else None
        )
    if left_result is not None and not bool(
        numpy.all(numpy.isfinite(left_result))
    ):
        if not (
            _comparable_finite_values(matrix_grad, numpy)
            and _comparable_finite_values(right_matrix, numpy)
        ):
            return None
        left_result = _scaled_matmul(
            matrix_grad,
            numpy.swapaxes(right_matrix, -1, -2),
            numpy,
        )
    if right_result is not None and not bool(
        numpy.all(numpy.isfinite(right_result))
    ):
        if not (
            _comparable_finite_values(left_matrix, numpy)
            and _comparable_finite_values(matrix_grad, numpy)
        ):
            return None
        right_result = _scaled_matmul(
            numpy.swapaxes(left_matrix, -1, -2),
            matrix_grad,
            numpy,
        )
    for result in (left_result, right_result):
        if result is not None and bool(numpy.any(numpy.isnan(result))):
            return None

    left_storage = None
    if left_result is not None:
        left_shape = (1, left.shape[0]) if left_vector else left.shape
        left_result = _reduce_matrix_gradient(left_result, left_shape, numpy)
        if left_result is None:
            return None
        if left_vector:
            left_result = left_result.reshape(left.shape)
        left_storage = _storage(
            left_result,
            dtype=grad.dtype,
            output_shape=left.shape,
            numpy=numpy,
        )
        if left_storage is None:
            return None
    right_storage = None
    if right_result is not None:
        right_shape = (right.shape[0], 1) if right_vector else right.shape
        right_result = _reduce_matrix_gradient(right_result, right_shape, numpy)
        if right_result is None:
            return None
        if right_vector:
            right_result = right_result.reshape(right.shape)
        right_storage = _storage(
            right_result,
            dtype=grad.dtype,
            output_shape=right.shape,
            numpy=numpy,
        )
        if right_storage is None:
            return None
    return left_storage, right_storage

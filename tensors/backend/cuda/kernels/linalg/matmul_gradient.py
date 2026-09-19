"""CuPy implementation of the matrix product VJP."""

from __future__ import annotations
import cupy
from typing import TYPE_CHECKING
from tensors.backend.storage import Storage
from tensors.backend.cuda.conversion import _errstate
from tensors.backend.cuda.conversion import _finite_operands
from tensors.backend.cuda.conversion import _storage
from tensors.backend.cuda.conversion import _view
from tensors.backend.cuda.conversion import _working_values

if TYPE_CHECKING:
    from tensors.tensor import Tensor
from tensors.backend.cuda.kernels.linalg.matmul_ops import _comparable_finite_values
from tensors.backend.cuda.kernels.linalg.matmul_ops import _matrix_gradient_view
from tensors.backend.cuda.kernels.linalg.matmul_ops import _matrix_view
from tensors.backend.cuda.kernels.linalg.matmul_ops import _reduce_matrix_gradient
from tensors.backend.cuda.kernels.linalg.matmul_ops import _scaled_matmul


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
    left_vector = left.ndim == 1
    right_vector = right.ndim == 1
    try:
        upstream = _working_values(grad)
        left_values = _working_values(left)
        right_values = _working_values(right)
    except ValueError:
        return None
    if not _finite_operands(upstream, left_values, right_values):
        return None
    left_matrix = _matrix_view(left_values, left_vector, left=True)
    right_matrix = _matrix_view(right_values, right_vector, left=False)
    matrix_grad = _matrix_gradient_view(upstream, left_vector, right_vector)
    need_left, need_right = needs_input_grad
    with _errstate(over="ignore", under="ignore", invalid="ignore"):
        left_result = (
            cupy.matmul(matrix_grad, cupy.swapaxes(right_matrix, -1, -2))
            if need_left
            else None
        )
        right_result = (
            cupy.matmul(cupy.swapaxes(left_matrix, -1, -2), matrix_grad)
            if need_right
            else None
        )
    if left_result is not None and (not bool(cupy.all(cupy.isfinite(left_result)))):
        if not (
            _comparable_finite_values(matrix_grad)
            and _comparable_finite_values(right_matrix)
        ):
            return None
        left_result = _scaled_matmul(matrix_grad, cupy.swapaxes(right_matrix, -1, -2))
    if right_result is not None and (not bool(cupy.all(cupy.isfinite(right_result)))):
        if not (
            _comparable_finite_values(left_matrix)
            and _comparable_finite_values(matrix_grad)
        ):
            return None
        right_result = _scaled_matmul(cupy.swapaxes(left_matrix, -1, -2), matrix_grad)
    for result in (left_result, right_result):
        if result is not None and bool(cupy.any(cupy.isnan(result))):
            return None
    left_storage = None
    if left_result is not None:
        left_shape = (1, left.shape[0]) if left_vector else left.shape
        left_result = _reduce_matrix_gradient(left_result, left_shape)
        if left_result is None:
            return None
        if left_vector:
            left_result = left_result.reshape(left.shape)
        left_storage = _storage(left_result, dtype=grad.dtype, output_shape=left.shape)
        if left_storage is None:
            return None
    right_storage = None
    if right_result is not None:
        right_shape = (right.shape[0], 1) if right_vector else right.shape
        right_result = _reduce_matrix_gradient(right_result, right_shape)
        if right_result is None:
            return None
        if right_vector:
            right_result = right_result.reshape(right.shape)
        right_storage = _storage(
            right_result, dtype=grad.dtype, output_shape=right.shape
        )
        if right_storage is None:
            return None
    return (left_storage, right_storage)

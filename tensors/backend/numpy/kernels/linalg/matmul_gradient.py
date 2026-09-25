"""NumPy implementation of matrix-product VJPs."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy

from tensors.backend.numpy.conversion import _storage
from tensors.backend.numpy.kernels.linalg.contraction import certified_matmul
from tensors.backend.numpy.kernels.reductions.sum_to_shape import sum_to_shape

if TYPE_CHECKING:
    from tensors.backend.storage import Storage
    from tensors.backend.types import MatmulMetadata
    from tensors.dtype import DataType


def matmul_gradient(
    grad_values: Any,
    left_values: Any,
    right_values: Any,
    *,
    metadata: MatmulMetadata,
    left_shape: tuple[int, ...],
    right_shape: tuple[int, ...],
    dtype: DataType,
    needs_input_grad: tuple[bool, ...] = (True, True),
) -> tuple[Storage | None, Storage | None] | None:
    """Execute requested certified matmul VJPs with NumPy-native values."""
    if dtype.kind != "floating":
        return None
    left_vector, right_vector = metadata[:2]
    try:
        upstream = grad_values.astype(numpy.float64, copy=False)
        left = left_values.astype(numpy.float64, copy=False)
        right = right_values.astype(numpy.float64, copy=False)
    except (TypeError, ValueError):
        return None
    left_matrix = left.reshape((1, left.shape[0])) if left_vector else left
    right_matrix = right.reshape((right.shape[0], 1)) if right_vector else right
    if left_vector and right_vector:
        matrix_grad = upstream.reshape((1, 1))
    elif left_vector:
        matrix_grad = upstream.reshape(upstream.shape[:-1] + (1, upstream.shape[-1]))
    elif right_vector:
        matrix_grad = upstream.reshape(upstream.shape + (1,))
    else:
        matrix_grad = upstream
    need_left, need_right = needs_input_grad
    left_result = (
        certified_matmul(matrix_grad, numpy.swapaxes(right_matrix, -1, -2))
        if need_left
        else None
    )
    right_result = (
        certified_matmul(numpy.swapaxes(left_matrix, -1, -2), matrix_grad)
        if need_right
        else None
    )
    if (need_left and left_result is None) or (need_right and right_result is None):
        return None
    left_storage = None
    if left_result is not None:
        left_result = numpy.where(left_result == 0.0, 0.0, left_result)
        if left_vector:
            left_result = numpy.squeeze(left_result, axis=-2)
        if tuple(left_result.shape) == left_shape:
            left_storage = _storage(left_result, dtype=dtype, output_shape=left_shape)
        else:
            left_storage = sum_to_shape(
                left_result,
                tuple(left_result.shape),
                left_shape,
                dtype=dtype,
            )
        if left_storage is None:
            return None
    right_storage = None
    if right_result is not None:
        right_result = numpy.where(right_result == 0.0, 0.0, right_result)
        if right_vector:
            right_result = numpy.squeeze(right_result, axis=-1)
        if tuple(right_result.shape) == right_shape:
            right_storage = _storage(
                right_result, dtype=dtype, output_shape=right_shape
            )
        else:
            right_storage = sum_to_shape(
                right_result,
                tuple(right_result.shape),
                right_shape,
                dtype=dtype,
            )
        if right_storage is None:
            return None
    return (left_storage, right_storage)

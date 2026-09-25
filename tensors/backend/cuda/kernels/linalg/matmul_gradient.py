"""CUDA implementation of matrix-product VJPs."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import cupy

from tensors.backend.cuda.conversion import _errstate, _storage, _widen
from tensors.backend.cuda.kernels.reductions.sum_to_shape import sum_to_shape

if TYPE_CHECKING:
    from tensors.backend.storage import Storage
    from tensors.dtype import DataType
    from tensors.backend.types import MatmulMetadata


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
    """Execute requested matmul VJPs with CUDA-native values."""
    if dtype.kind != "floating":
        return None
    left_vector, right_vector = metadata[:2]
    try:
        upstream = _widen(grad_values)
        left = _widen(left_values)
        right = _widen(right_values)
    except (TypeError, ValueError):
        return None
    if not bool(
        cupy.all(cupy.isfinite(upstream))
        & cupy.all(cupy.isfinite(left))
        & cupy.all(cupy.isfinite(right))
    ):
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
    if left_result is not None and bool(cupy.any(~cupy.isfinite(left_result))):
        return None
    if right_result is not None and bool(cupy.any(~cupy.isfinite(right_result))):
        return None
    left_storage = None
    if left_result is not None:
        left_result = cupy.where(left_result == 0.0, 0.0, left_result)
        if left_vector:
            left_result = cupy.squeeze(left_result, axis=-2)
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
        right_result = cupy.where(right_result == 0.0, 0.0, right_result)
        if right_vector:
            right_result = cupy.squeeze(right_result, axis=-1)
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

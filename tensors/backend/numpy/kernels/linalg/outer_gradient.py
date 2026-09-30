"""NumPy implementation of outer-product VJPs."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy

from tensors.backend.numpy.conversion import _storage
from tensors.backend.numpy.kernels.linalg.contraction import pairwise_matmul
from tensors.backend.numpy.kernels.reductions.pairwise import to_declared_dtype

if TYPE_CHECKING:
    from tensors.backend.storage import Storage
    from tensors.dtype import DataType


def outer_gradient(
    grad_values: Any,
    left_values: Any,
    right_values: Any,
    *,
    left_shape: tuple[int, ...],
    right_shape: tuple[int, ...],
    dtype: DataType,
    needs_input_grad: tuple[bool, ...] = (True, True),
) -> tuple[Storage | None, Storage | None] | None:
    """Execute requested outer-product VJPs as products then pairwise sums."""
    if dtype.kind != "floating":
        return None
    try:
        upstream = to_declared_dtype(grad_values, dtype)
        left = to_declared_dtype(left_values, dtype)
        right = to_declared_dtype(right_values, dtype)
    except (TypeError, ValueError):
        return None
    need_left, need_right = needs_input_grad
    left_result = (
        pairwise_matmul(upstream, right.reshape((right.shape[0], 1)))
        if need_left
        else None
    )
    right_result = (
        pairwise_matmul(left.reshape((1, left.shape[0])), upstream)
        if need_right
        else None
    )
    if left_result is not None:
        left_result = numpy.squeeze(left_result, axis=-1)
    if right_result is not None:
        right_result = numpy.squeeze(right_result, axis=-2)
    left_storage = (
        _storage(
            left_result,
            dtype=dtype,
            output_shape=left_shape,
        )
        if left_result is not None
        else None
    )
    right_storage = (
        _storage(
            right_result,
            dtype=dtype,
            output_shape=right_shape,
        )
        if right_result is not None
        else None
    )
    if (need_left and left_storage is None) or (need_right and right_storage is None):
        return None
    return (left_storage, right_storage)

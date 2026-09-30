"""CUDA implementation of matrix products."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import cupy

from tensors.backend.cuda.conversion import _storage
from tensors.backend.cuda.kernels.linalg.contraction import pairwise_matmul
from tensors.backend.cuda.kernels.reductions.pairwise import to_declared_dtype

if TYPE_CHECKING:
    from tensors.backend.storage import Storage
    from tensors.backend.types import MatmulMetadata
    from tensors.dtype import DataType


def matmul(
    left_values: Any,
    right_values: Any,
    *,
    metadata: MatmulMetadata,
    dtype: DataType,
    output_shape: tuple[int, ...],
) -> Storage | None:
    """Execute a floating matrix product as products then a pairwise sum."""
    if dtype.kind != "floating":
        return None
    try:
        left = to_declared_dtype(left_values, dtype)
        right = to_declared_dtype(right_values, dtype)
        left_vector, right_vector = metadata[:2]
        left_matrix = left.reshape((1, left.shape[0])) if left_vector else left
        right_matrix = right.reshape((right.shape[0], 1)) if right_vector else right
        result = pairwise_matmul(left_matrix, right_matrix)
    except (TypeError, ValueError):
        return None
    if left_vector:
        result = cupy.squeeze(result, axis=-2)
    if right_vector:
        result = cupy.squeeze(result, axis=-1)
    return _storage(result, dtype=dtype, output_shape=output_shape)

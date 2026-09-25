"""NumPy implementation of matrix products."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy

from tensors.backend.numpy.conversion import _storage
from tensors.backend.numpy.kernels.linalg.contraction import certified_matmul

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
    """Execute a certified floating matrix product with NumPy-native values."""
    if dtype.kind != "floating":
        return None
    try:
        left = left_values.astype(numpy.float64, copy=False)
        right = right_values.astype(numpy.float64, copy=False)
        left_vector, right_vector = metadata[:2]
        left_matrix = left.reshape((1, left.shape[0])) if left_vector else left
        right_matrix = right.reshape((right.shape[0], 1)) if right_vector else right
        result = certified_matmul(left_matrix, right_matrix)
    except (TypeError, ValueError):
        return None
    if result is None:
        return None
    if left_vector:
        result = numpy.squeeze(result, axis=-2)
    if right_vector:
        result = numpy.squeeze(result, axis=-1)
    result = numpy.where(result == 0.0, 0.0, result)
    return _storage(result, dtype=dtype, output_shape=output_shape)

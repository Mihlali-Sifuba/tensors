"""NumPy implementation of matrix products."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy

from tensors.backend.numpy.conversion import _errstate, _storage

if TYPE_CHECKING:
    from tensors.backend.storage import Storage
    from tensors.dtype import DataType
    from tensors.backend.types import MatmulMetadata


def matmul(
    left_values: Any,
    right_values: Any,
    *,
    metadata: MatmulMetadata,
    dtype: DataType,
    output_shape: tuple[int, ...],
) -> Storage | None:
    """Execute a floating matrix product with NumPy-native values."""
    if dtype.kind != "floating":
        return None
    try:
        left = left_values.astype(numpy.float64, copy=False)
        right = right_values.astype(numpy.float64, copy=False)
        with _errstate(over="ignore", under="ignore", invalid="ignore"):
            result = numpy.matmul(left, right)
    except (TypeError, ValueError):
        return None
    finite_operands = numpy.all(numpy.isfinite(left)) & numpy.all(numpy.isfinite(right))
    if bool(finite_operands & numpy.any(~numpy.isfinite(result))):
        return None
    result = numpy.where(result == 0.0, 0.0, result)
    return _storage(result, dtype=dtype, output_shape=output_shape)

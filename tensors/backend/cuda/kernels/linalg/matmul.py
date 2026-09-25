"""CUDA implementation of matrix products."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import cupy

from tensors.backend.cuda.conversion import _errstate, _storage, _widen

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
    """Execute a floating matrix product with CUDA-native values."""
    if dtype.kind != "floating":
        return None
    try:
        left = _widen(left_values)
        right = _widen(right_values)
        with _errstate(over="ignore", under="ignore", invalid="ignore"):
            result = cupy.matmul(left, right)
    except (TypeError, ValueError):
        return None
    finite_operands = cupy.all(cupy.isfinite(left)) & cupy.all(cupy.isfinite(right))
    if bool(finite_operands & cupy.any(~cupy.isfinite(result))):
        return None
    result = cupy.where(result == 0.0, 0.0, result)
    return _storage(result, dtype=dtype, output_shape=output_shape)

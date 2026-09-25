"""CUDA implementation of vector outer products."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import cupy

from tensors.backend.cuda.conversion import _errstate, _storage, _widen

if TYPE_CHECKING:
    from tensors.backend.storage import Storage
    from tensors.dtype import DataType


def outer(
    left_values: Any,
    right_values: Any,
    *,
    dtype: DataType,
    output_shape: tuple[int, ...],
) -> Storage | None:
    """Execute a floating outer product with CUDA-native values."""
    if dtype.kind != "floating":
        return None
    try:
        left = _widen(left_values)
        right = _widen(right_values)
        with _errstate(over="ignore", under="ignore", invalid="ignore"):
            result = cupy.multiply.outer(left, right)
    except (TypeError, ValueError):
        return None
    return _storage(result, dtype=dtype, output_shape=output_shape)

"""CuPy implementation of the Euclidean norm."""

from __future__ import annotations
import cupy
from typing import Any, TYPE_CHECKING
from tensors.backend.storage import Storage
from tensors.backend.cuda.conversion import _errstate
from tensors.backend.cuda.conversion import _arithmetic_storage as _storage
from tensors.backend.cuda.conversion import _widen

if TYPE_CHECKING:
    from tensors.dtype import DataType


def reduce_norm(
    values: Any,
    input_shape: tuple[int, ...],
    axes: tuple[int, ...],
    *,
    keepdims: bool,
    dtype: DataType,
    output_shape: tuple[int, ...],
) -> Storage | None:
    """Run a numerically guarded NumPy reduction."""
    if values.size == 0:
        return _storage(
            cupy.full(output_shape, 0.0), dtype=dtype, output_shape=output_shape
        )
    axis = axes
    values = _widen(values)
    with _errstate(over="ignore", under="ignore", invalid="ignore"):
        absolute = cupy.abs(values)
        scale = cupy.max(absolute, axis=axis, keepdims=True)
        safe_scale = cupy.where(scale == 0.0, 1.0, scale)
        normalized = values / safe_scale
        normalized_magnitude = cupy.sqrt(
            cupy.sum(normalized * normalized, axis=axis, keepdims=keepdims)
        )
        output_scale = scale if keepdims else cupy.squeeze(scale, axis=axis)
        result = output_scale * normalized_magnitude
    has_nan = cupy.any(cupy.isnan(values), axis=axis, keepdims=keepdims)
    has_infinity = cupy.any(cupy.isinf(values), axis=axis, keepdims=keepdims)
    result = cupy.where(has_nan, cupy.nan, cupy.where(has_infinity, cupy.inf, result))
    return _storage(result, dtype=dtype, output_shape=output_shape)

"""CuPy implementation of the standard deviation."""

from __future__ import annotations
import cupy
from typing import Any, TYPE_CHECKING
from tensors.backend.storage import Storage
from tensors.backend.cuda.conversion import _errstate
from tensors.backend.cuda.conversion import _arithmetic_storage as _storage
from tensors.backend.cuda.conversion import _widen

if TYPE_CHECKING:
    from tensors.dtype import DataType


def reduce_std(
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
            cupy.full(output_shape, cupy.nan), dtype=dtype, output_shape=output_shape
        )
    axis = axes
    values = _widen(values)
    with _errstate(over="ignore", under="ignore", invalid="ignore"):
        scale = cupy.max(cupy.abs(values), axis=axis, keepdims=True)
        safe_scale = cupy.where(scale == 0.0, 1.0, scale)
        normalized_values = values / safe_scale
        center = cupy.mean(normalized_values, axis=axis, keepdims=True)
        normalized = normalized_values - center
        normalized_variance = cupy.mean(
            normalized * normalized, axis=axis, keepdims=keepdims
        )
        output_scale = scale if keepdims else cupy.squeeze(scale, axis=axis)
        deviation = output_scale * cupy.sqrt(normalized_variance)
        result = deviation
    return _storage(result, dtype=dtype, output_shape=output_shape)

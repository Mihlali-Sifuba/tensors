"""CuPy implementation of the standard deviation VJP."""

from __future__ import annotations
import cupy
from typing import Any, TYPE_CHECKING
from tensors.backend.storage import Storage
from tensors.dtype import DataType
from tensors.backend.cuda.conversion import _errstate
from tensors.backend.cuda.conversion import _arithmetic_storage as _storage
from tensors.backend.cuda.conversion import _widen


def reduce_std_gradient(
    upstream: Any,
    values: Any,
    input_shape: tuple[int, ...],
    axes: tuple[int, ...],
    *,
    keepdims: bool,
    dtype: DataType,
) -> Storage | None:
    """Run fused VJPs for reductions with regular native fast paths."""
    values = _widen(values)
    if values.size == 0:
        return _storage(values, dtype=dtype, output_shape=input_shape)
    upstream = _widen(upstream)
    expanded_shape = tuple(
        (1 if dimension in axes else size for dimension, size in enumerate(input_shape))
    )
    try:
        expanded = upstream.reshape(expanded_shape)
    except ValueError:
        return None
    count = 1
    for axis in axes:
        count *= input_shape[axis]
    with _errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore"):
        scale = cupy.max(cupy.abs(values), axis=axes, keepdims=True)
        safe_scale = cupy.where(scale == 0.0, 1.0, scale)
        normalized = values / safe_scale
        center = cupy.mean(normalized, axis=axes, keepdims=True)
        centered = normalized - center
        deviation = cupy.sqrt(cupy.mean(centered * centered, axis=axes, keepdims=True))
        derivative = cupy.where(deviation == 0.0, 0.0, centered / (count * deviation))
        result = expanded * derivative
    return _storage(result, dtype=dtype, output_shape=input_shape)

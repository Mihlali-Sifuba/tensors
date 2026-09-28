"""CuPy implementation of the Euclidean norm VJP."""

from __future__ import annotations
import cupy
from typing import Any, TYPE_CHECKING
from tensors.backend.storage import Storage
from tensors.dtype import DataType
from tensors.backend.cuda.conversion import _errstate
from tensors.backend.cuda.conversion import _arithmetic_storage as _storage
from tensors.backend.cuda.conversion import _widen


def reduce_norm_gradient(
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
    expanded = upstream.reshape(expanded_shape)
    with _errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore"):
        scale = cupy.max(cupy.abs(values), axis=axes, keepdims=True)
        safe_scale = cupy.where(scale == 0.0, 1.0, scale)
        normalized = values / safe_scale
        magnitude = cupy.sqrt(
            cupy.sum(normalized * normalized, axis=axes, keepdims=True)
        )
        derivative = cupy.where(magnitude == 0.0, 0.0, normalized / magnitude)
        result = expanded * derivative
    return _storage(result, dtype=dtype, output_shape=input_shape)

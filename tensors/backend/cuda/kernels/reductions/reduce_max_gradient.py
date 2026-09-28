"""CuPy implementation of the maximum VJP."""

from __future__ import annotations
import cupy
from typing import Any, TYPE_CHECKING
from tensors.backend.storage import Storage
from tensors.dtype import DataType
from tensors.backend.cuda.conversion import _errstate
from tensors.backend.cuda.conversion import _arithmetic_storage as _storage
from tensors.backend.cuda.conversion import _widen


def reduce_max_gradient(
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
        has_nan = cupy.any(cupy.isnan(values), axis=axes, keepdims=True)
        function = cupy.max
        extreme = function(values, axis=axes, keepdims=True)
        selected = values == extreme
        ties = cupy.sum(selected, axis=axes, keepdims=True)
        result = cupy.where(has_nan, cupy.nan, expanded * selected / ties)
    return _storage(result, dtype=dtype, output_shape=input_shape)

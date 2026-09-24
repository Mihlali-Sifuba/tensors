"""CuPy implementation of summation."""

from __future__ import annotations
import cupy
from typing import Any, TYPE_CHECKING
from tensors.backend.storage import Storage
from tensors.backend.cuda.conversion import _errstate
from tensors.backend.cuda.conversion import _arithmetic_storage as _storage
from tensors.backend.cuda.conversion import _widen
from tensors.backend.cuda.kernels.reductions.stability import _scaled_sum
from tensors.backend.cuda.kernels.reductions.stability import _summation_guard

if TYPE_CHECKING:
    from tensors.dtype import DataType


def reduce_sum(
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
            cupy.full(output_shape, 0), dtype=dtype, output_shape=output_shape
        )
    axis = axes
    values = _widen(values)
    if dtype.kind == "integer":
        result = cupy.sum(values, axis=axis, keepdims=keepdims, dtype=cupy.int64)
        return _storage(result, dtype=dtype, output_shape=output_shape)
    with _errstate(over="ignore", under="ignore", invalid="ignore"):
        direct = cupy.sum(values, axis=axis, keepdims=True)
    ordinary = False
    if ordinary:
        result = direct
    else:
        with _errstate(over="ignore", under="ignore", invalid="ignore"):
            scaled = _scaled_sum(values, axis)
        safe = _summation_guard(values, axes=axis, keepdims=True)
        direct_safe = safe & cupy.all(cupy.isfinite(direct))
        result = cupy.where(direct_safe, direct, scaled)
        finite_group = cupy.all(cupy.isfinite(values), axis=axis, keepdims=True)
        result = cupy.where(finite_group, result, direct)
    if not keepdims and axis:
        result = cupy.squeeze(result, axis=axis)
    return _storage(result, dtype=dtype, output_shape=output_shape)

"""CuPy implementation of summation."""

from __future__ import annotations
import cupy
from typing import TYPE_CHECKING
from tensors.backend.storage import Storage
from tensors.backend.cuda.conversion import _errstate
from tensors.backend.cuda.conversion import _storage
from tensors.backend.cuda.conversion import _view
from tensors.backend.cuda.kernels.reductions.stability import _scaled_sum
from tensors.backend.cuda.kernels.reductions.stability import _summation_guard

if TYPE_CHECKING:
    from tensors.dtype import DataType
    from tensors.tensor import Tensor


def reduce_sum(
    value: Tensor,
    axes: tuple[int, ...],
    *,
    keepdims: bool,
    dtype: DataType,
    output_shape: tuple[int, ...],
) -> Storage | None:
    """Run a numerically guarded NumPy reduction."""
    if value.size == 0:
        return None
    axis = axes
    values = _view(value).astype(cupy.float64, copy=False)
    if dtype.kind == "integer":
        return None
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
        valid = safe & ~cupy.any(cupy.isnan(result))
        if not bool(valid):
            return None
    if not keepdims and axis:
        result = cupy.squeeze(result, axis=axis)
    return _storage(result, dtype=dtype, output_shape=output_shape)

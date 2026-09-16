"""CuPy implementation of the variance."""

from __future__ import annotations
import cupy
from typing import TYPE_CHECKING
from tensors.backend.storage import Storage
from tensors.backend.cuda.conversion import _errstate
from tensors.backend.cuda.conversion import _storage
from tensors.backend.cuda.conversion import _view

if TYPE_CHECKING:
    from tensors.dtype import DataType
    from tensors.tensor import Tensor


def reduce_variance(
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
    with _errstate(over="ignore", under="ignore", invalid="ignore"):
        center = cupy.mean(values, axis=axis, keepdims=True)
        centered = values - center
        scale = cupy.max(cupy.abs(centered), axis=axis, keepdims=True)
        safe_scale = cupy.where(scale == 0.0, 1.0, scale)
        normalized = centered / safe_scale
        normalized_variance = cupy.mean(
            normalized * normalized, axis=axis, keepdims=keepdims
        )
        output_scale = scale if keepdims else cupy.squeeze(scale, axis=axis)
        deviation = output_scale * cupy.sqrt(normalized_variance)
        result = deviation * deviation
    valid = (
        cupy.all(cupy.isfinite(values))
        & cupy.all(cupy.isfinite(centered))
        & cupy.all(cupy.isfinite(result))
    )
    if not bool(valid):
        return None
    return _storage(result, dtype=dtype, output_shape=output_shape)

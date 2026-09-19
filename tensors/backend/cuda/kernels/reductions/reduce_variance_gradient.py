"""CuPy implementation of the variance VJP."""

from __future__ import annotations
import cupy
from typing import TYPE_CHECKING
from tensors.backend.storage import Storage
from tensors.backend.cuda.conversion import _errstate
from tensors.backend.cuda.conversion import _storage
from tensors.backend.cuda.conversion import _working_values

if TYPE_CHECKING:
    from tensors.tensor import Tensor


def reduce_variance_gradient(
    grad: Tensor, value: Tensor, axes: tuple[int, ...], *, keepdims: bool
) -> Storage | None:
    """Run fused VJPs for reductions with regular native fast paths."""
    values = _working_values(value)
    upstream = _working_values(grad)
    expanded_shape = tuple(
        (1 if dimension in axes else size for dimension, size in enumerate(value.shape))
    )
    try:
        expanded = upstream.reshape(expanded_shape)
    except ValueError:
        return None
    count = 1
    for axis in axes:
        count *= value.shape[axis]
    with _errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore"):
        if count == 0:
            return None
        center = cupy.mean(values, axis=axes, keepdims=True)
        centered = values - center
        scale = cupy.max(cupy.abs(centered), axis=axes, keepdims=True)
        safe_scale = cupy.where(scale == 0.0, 1.0, scale)
        normalized = centered / safe_scale
        result = expanded * normalized * scale * (2.0 / count)
        valid = (
            cupy.all(cupy.isfinite(values))
            & cupy.all(cupy.isfinite(centered))
            & cupy.all(cupy.isfinite(result))
        )
        if not bool(valid):
            return None
    return _storage(result, dtype=grad.dtype, output_shape=value.shape)

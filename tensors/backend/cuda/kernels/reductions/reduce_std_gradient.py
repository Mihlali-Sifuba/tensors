"""CuPy implementation of the standard deviation VJP."""

from __future__ import annotations
import cupy
from typing import TYPE_CHECKING
from tensors.backend.storage import Storage
from tensors.backend.cuda.conversion import _errstate
from tensors.backend.cuda.conversion import _storage
from tensors.backend.cuda.conversion import _working_values

if TYPE_CHECKING:
    from tensors.tensor import Tensor


def reduce_std_gradient(
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
        if not bool(cupy.all(cupy.isfinite(values))) or count == 0:
            return None
        scale = cupy.max(cupy.abs(values), axis=axes, keepdims=True)
        safe_scale = cupy.where(scale == 0.0, 1.0, scale)
        normalized = values / safe_scale
        center = cupy.mean(normalized, axis=axes, keepdims=True)
        centered = normalized - center
        deviation = cupy.sqrt(cupy.mean(centered * centered, axis=axes, keepdims=True))
        derivative = cupy.where(deviation == 0.0, 0.0, centered / (count * deviation))
        result = expanded * derivative
    return _storage(result, dtype=grad.dtype, output_shape=value.shape)

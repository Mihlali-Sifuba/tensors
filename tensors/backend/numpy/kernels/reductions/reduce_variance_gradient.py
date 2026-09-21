"""NumPy implementation of the variance VJP."""

from __future__ import annotations
import numpy
from typing import TYPE_CHECKING
from tensors.backend.storage import Storage
from tensors.backend.numpy.conversion import _errstate
from tensors.backend.numpy.conversion import _storage
from tensors.backend.numpy.conversion import tensor_to_logical_array

if TYPE_CHECKING:
    from tensors.tensor import Tensor


def reduce_variance_gradient(
    grad: Tensor, value: Tensor, axes: tuple[int, ...], *, keepdims: bool
) -> Storage | None:
    """Run fused VJPs for reductions with regular native fast paths."""
    values = tensor_to_logical_array(value).astype(numpy.float64, copy=False)
    upstream = tensor_to_logical_array(grad).astype(numpy.float64, copy=False)
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
        center = numpy.mean(values, axis=axes, keepdims=True)
        centered = values - center
        scale = numpy.max(numpy.abs(centered), axis=axes, keepdims=True)
        safe_scale = numpy.where(scale == 0.0, 1.0, scale)
        normalized = centered / safe_scale
        result = expanded * normalized * scale * (2.0 / count)
        valid = (
            numpy.all(numpy.isfinite(values))
            & numpy.all(numpy.isfinite(centered))
            & numpy.all(numpy.isfinite(result))
        )
        if not bool(valid):
            return None
    return _storage(result, dtype=grad.dtype, output_shape=value.shape)

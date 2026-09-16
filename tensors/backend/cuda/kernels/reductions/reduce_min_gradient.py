"""CuPy implementation of the minimum VJP."""

from __future__ import annotations
import cupy
from typing import TYPE_CHECKING
from tensors.backend.storage import Storage
from tensors.backend.cuda.conversion import _errstate
from tensors.backend.cuda.conversion import _storage
from tensors.backend.cuda.conversion import _view

if TYPE_CHECKING:
    from tensors.tensor import Tensor


def reduce_min_gradient(
    grad: Tensor, value: Tensor, axes: tuple[int, ...], *, keepdims: bool
) -> Storage | None:
    """Run fused VJPs for reductions with regular native fast paths."""
    values = _view(value).astype(cupy.float64, copy=False)
    upstream = _view(grad).astype(cupy.float64, copy=False)
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
        has_nan = cupy.any(cupy.isnan(values), axis=axes, keepdims=True)
        function = cupy.min
        extreme = function(values, axis=axes, keepdims=True)
        selected = values == extreme
        ties = cupy.sum(selected, axis=axes, keepdims=True)
        result = cupy.where(has_nan, cupy.nan, expanded * selected / ties)
    return _storage(result, dtype=grad.dtype, output_shape=value.shape)

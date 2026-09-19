"""NumPy implementation of summation down to a broadcast shape."""

from __future__ import annotations
import numpy
from typing import TYPE_CHECKING
from tensors.backend.storage import Storage
from tensors.backend.numpy.conversion import _storage
from tensors.backend.numpy.conversion import _view
from tensors.backend.numpy.kernels.reductions.stability import _scaled_sum
from tensors.backend.numpy.kernels.reductions.stability import _stable_sum_candidate
from tensors.backend.numpy.kernels.reductions.stability import _sum_axes

if TYPE_CHECKING:
    from tensors.tensor import Tensor


def sum_to_shape(gradient: Tensor, shape: tuple[int, ...]) -> Storage | None:
    """Reduce a broadcast gradient using guarded native summation."""
    if gradient.dtype.kind == "integer":
        return None
    layout = _sum_axes(gradient.shape, shape)
    if layout is None:
        return None
    _, axes = layout
    values = _view(gradient).astype(numpy.float64, copy=False)
    safe = _stable_sum_candidate(values, axes)
    result = _scaled_sum(values, axes)
    valid = safe & ~numpy.any(numpy.isnan(result))
    if not bool(valid):
        return None
    return _storage(
        numpy.asarray(result).reshape(shape), dtype=gradient.dtype, output_shape=shape
    )

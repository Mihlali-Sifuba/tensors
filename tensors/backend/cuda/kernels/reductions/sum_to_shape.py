"""CuPy implementation of summation down to a broadcast shape."""

from __future__ import annotations
import cupy
from typing import TYPE_CHECKING
from tensors.backend.storage import Storage
from tensors.backend.cuda.conversion import _storage
from tensors.backend.cuda.conversion import _working_values
from tensors.backend.cuda.kernels.reductions.stability import _scaled_sum
from tensors.backend.cuda.kernels.reductions.stability import _stable_sum_candidate
from tensors.backend.cuda.kernels.reductions.stability import _sum_axes

if TYPE_CHECKING:
    from tensors.tensor import Tensor


def sum_to_shape(gradient: Tensor, shape: tuple[int, ...]) -> Storage | None:
    """Reduce a broadcast gradient using guarded native summation."""
    if gradient.dtype.kind == "integer":
        return None
    axes = _sum_axes(gradient.shape, shape)
    if axes is None:
        return None
    values = _working_values(gradient)
    safe = _stable_sum_candidate(values, axes)
    result = _scaled_sum(values, axes)
    valid = safe & ~cupy.any(cupy.isnan(result))
    if not bool(valid):
        return None
    return _storage(
        cupy.asarray(result).reshape(shape), dtype=gradient.dtype, output_shape=shape
    )

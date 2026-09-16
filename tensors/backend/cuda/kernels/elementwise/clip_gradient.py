"""CuPy implementation of the clipping VJP."""

from __future__ import annotations
import cupy
from typing import TYPE_CHECKING
from tensors.backend.storage import Storage
from tensors.backend.cuda.conversion import _storage
from tensors.backend.cuda.conversion import _view

if TYPE_CHECKING:
    from tensors.tensor import Tensor


def clip_gradient(
    grad: Tensor,
    value: Tensor,
    min_value: int | float | None,
    max_value: int | float | None,
) -> Storage | None:
    """Run the clipping VJP with zero boundary subgradients."""
    values = _view(value).astype(cupy.float64, copy=False)
    upstream = _view(grad).astype(cupy.float64, copy=False)
    mask = cupy.ones(value.shape, dtype=bool)
    if min_value is not None:
        mask &= values > min_value
    if max_value is not None:
        mask &= values < max_value
    result = cupy.where(cupy.isnan(values), cupy.nan, cupy.where(mask, upstream, 0.0))
    return _storage(result, dtype=grad.dtype, output_shape=value.shape)

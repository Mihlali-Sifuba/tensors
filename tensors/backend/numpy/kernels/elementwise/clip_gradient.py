"""NumPy implementation of the clipping VJP."""

from __future__ import annotations
import numpy
from typing import TYPE_CHECKING
from tensors.backend.storage import Storage
from tensors.backend.numpy.conversion import _storage
from tensors.backend.numpy.conversion import tensor_to_logical_array

if TYPE_CHECKING:
    from tensors.tensor import Tensor


def clip_gradient(
    grad: Tensor,
    value: Tensor,
    min_value: int | float | None,
    max_value: int | float | None,
) -> Storage | None:
    """Run the clipping VJP with zero boundary subgradients."""
    values = tensor_to_logical_array(value).astype(numpy.float64, copy=False)
    upstream = tensor_to_logical_array(grad).astype(numpy.float64, copy=False)
    mask = numpy.ones(value.shape, dtype=bool)
    if min_value is not None:
        mask &= values > min_value
    if max_value is not None:
        mask &= values < max_value
    result = numpy.where(
        numpy.isnan(values), numpy.nan, numpy.where(mask, upstream, 0.0)
    )
    return _storage(result, dtype=grad.dtype, output_shape=value.shape)

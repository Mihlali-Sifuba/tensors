"""NumPy implementation of the arithmetic mean VJP."""

from __future__ import annotations
import numpy
from typing import TYPE_CHECKING
from tensors.backend.storage import Storage
from tensors.backend.numpy.conversion import _errstate
from tensors.backend.numpy.conversion import _storage
from tensors.backend.numpy.conversion import _view

if TYPE_CHECKING:
    from tensors.tensor import Tensor


def reduce_mean_gradient(
    grad: Tensor, value: Tensor, axes: tuple[int, ...], *, keepdims: bool
) -> Storage | None:
    """Broadcast the averaged upstream gradient back over the reduced axes."""
    upstream = _view(grad).astype(numpy.float64, copy=False)
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
        result = numpy.broadcast_to(expanded / count, value.shape)
    return _storage(result, dtype=grad.dtype, output_shape=value.shape)

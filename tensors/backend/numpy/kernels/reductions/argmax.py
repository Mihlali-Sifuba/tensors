"""NumPy implementation of the index of the maximum."""

from __future__ import annotations
import numpy
from typing import TYPE_CHECKING
from tensors.backend.storage import Storage
from tensors.backend.numpy.conversion import _storage
from tensors.backend.numpy.conversion import _view

if TYPE_CHECKING:
    from tensors.tensor import Tensor


def argmax(
    value: Tensor, axis: int | None, *, keepdims: bool, output_shape: tuple[int, ...]
) -> Storage | None:
    """Run a first-occurrence argmin or argmax reduction."""
    if value.size == 0:
        return None
    from tensors.dtype import int64

    values = _view(value)
    function = numpy.argmax
    result = function(values, axis=axis, keepdims=keepdims)
    return _storage(result, dtype=int64, output_shape=output_shape)

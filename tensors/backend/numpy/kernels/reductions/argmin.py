"""NumPy implementation of the index of the minimum."""

from __future__ import annotations
import numpy
from typing import Any, TYPE_CHECKING
from tensors.backend.storage import Storage
from tensors.backend.numpy.conversion import _arithmetic_storage as _storage


def argmin(
    values: Any,
    input_shape: tuple[int, ...],
    axis: int | None,
    *,
    keepdims: bool,
    output_shape: tuple[int, ...],
) -> Storage | None:
    """Run a first-occurrence argmin or argmax reduction."""
    if values.size == 0:
        raise ValueError("Cannot compute argmin of empty tensor")
    from tensors.dtype import int64

    function = numpy.argmin
    result = function(values, axis=axis, keepdims=keepdims)
    return _storage(result, dtype=int64, output_shape=output_shape)

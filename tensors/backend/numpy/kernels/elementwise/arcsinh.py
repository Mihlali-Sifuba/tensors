"""NumPy implementation of arcsinh."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy

from tensors.backend.numpy.conversion import _errstate
from tensors.backend.numpy.storage import NumPyStorage
from tensors.backend.storage import Storage

if TYPE_CHECKING:
    from tensors.dtype import DataType


def arcsinh(
    values: Any,
    *,
    dtype: DataType,
    output_shape: tuple[int, ...],
) -> Storage:
    """Evaluate arcsinh on prepared native values."""
    with _errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore"):
        working = numpy.asarray(values, dtype=numpy.float64)
        result = numpy.arcsinh(working)
        narrowed = result.astype(numpy.dtype(dtype.name), copy=False)
    storage = NumPyStorage(narrowed, dtype)
    if storage.size != math.prod(output_shape):
        raise RuntimeError("arcsinh kernel returned an unexpected result size")
    return storage

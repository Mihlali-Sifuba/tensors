"""NumPy implementation of addition."""

from __future__ import annotations
import numpy
from typing import TYPE_CHECKING, Any
from tensors.backend.storage import Storage
from tensors.backend.numpy.conversion import _errstate
from tensors.backend.numpy.conversion import _arithmetic_storage

if TYPE_CHECKING:
    from tensors.dtype import DataType


def add(
    left: Any,
    right: Any,
    *,
    dtype: DataType,
    output_shape: tuple[int, ...],
) -> Storage:
    """Return native storage at the declared dtype."""
    with _errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore"):
        result = numpy.add(left, right)
    return _arithmetic_storage(result, dtype=dtype, output_shape=output_shape)

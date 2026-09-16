"""NumPy implementation of the vector outer product."""

from __future__ import annotations
import numpy
from typing import TYPE_CHECKING
from tensors.backend.storage import Storage
from tensors.backend.numpy.conversion import _errstate
from tensors.backend.numpy.conversion import _operand
from tensors.backend.numpy.conversion import _storage

if TYPE_CHECKING:
    from tensors.dtype import DataType
    from tensors.tensor import Tensor


def outer(left: Tensor, right: Tensor, *, dtype: DataType) -> Storage | None:
    """Run a vector outer product."""
    try:
        left_values = _operand(left, dtype)
        right_values = _operand(right, dtype)
    except (TypeError, ValueError):
        return None
    with _errstate(over="ignore", under="ignore", invalid="ignore"):
        result = numpy.multiply.outer(left_values, right_values)
    return _storage(result, dtype=dtype, output_shape=(left.size, right.size))

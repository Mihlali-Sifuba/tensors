"""NumPy implementation of the elementwise minimum."""

from __future__ import annotations
import numpy
from typing import TYPE_CHECKING
from tensors.backend.storage import Storage
from tensors.backend.numpy.conversion import _storage
from tensors.backend.numpy.conversion import tensor_to_logical_array

if TYPE_CHECKING:
    from tensors.dtype import DataType
    from tensors.tensor import Tensor


def minimum(
    left: Tensor, right: Tensor, *, dtype: DataType, output_shape: tuple[int, ...]
) -> Storage | None:
    """Run a broadcasting elementwise minimum or maximum."""
    function = numpy.minimum
    try:
        result = function(tensor_to_logical_array(left), tensor_to_logical_array(right))
    except (TypeError, ValueError):
        return None
    return _storage(result, dtype=dtype, output_shape=output_shape)

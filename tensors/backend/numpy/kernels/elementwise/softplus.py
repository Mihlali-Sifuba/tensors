"""NumPy implementation of softplus."""

from __future__ import annotations
import numpy
from typing import TYPE_CHECKING
from tensors.backend.storage import Storage
from tensors.backend.numpy.conversion import _errstate
from tensors.backend.numpy.conversion import _storage
from tensors.backend.numpy.conversion import tensor_to_logical_array

if TYPE_CHECKING:
    from tensors.dtype import DataType
    from tensors.tensor import Tensor


def softplus(value: Tensor, *, dtype: DataType) -> Storage | None:
    """Apply softplus through log1p so a large input does not overflow."""
    if dtype.kind == "integer":
        return None
    try:
        values = tensor_to_logical_array(value).astype(numpy.float64, copy=False)
    except (TypeError, ValueError):
        return None
    with _errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore"):
        result = numpy.log1p(numpy.exp(-numpy.abs(values))) + numpy.maximum(values, 0.0)
    return _storage(result, dtype=dtype, output_shape=value.shape)

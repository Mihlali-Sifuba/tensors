"""NumPy implementation of the logistic function."""

from __future__ import annotations
import numpy
from typing import TYPE_CHECKING
from tensors.backend.storage import Storage
from tensors.backend.numpy.conversion import _errstate
from tensors.backend.numpy.conversion import _storage
from tensors.backend.numpy.conversion import _view

if TYPE_CHECKING:
    from tensors.dtype import DataType
    from tensors.tensor import Tensor


def sigmoid(value: Tensor, *, dtype: DataType) -> Storage | None:
    """Apply the logistic function from whichever side avoids overflow."""
    if dtype.kind == "integer":
        return None
    try:
        values = _view(value).astype(numpy.float64, copy=False)
    except (TypeError, ValueError):
        return None
    with _errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore"):
        magnitude = numpy.exp(-numpy.abs(values))
        result = numpy.where(
            values >= 0.0, 1.0 / (1.0 + magnitude), magnitude / (1.0 + magnitude)
        )
    return _storage(result, dtype=dtype, output_shape=value.shape)

"""NumPy implementation of dtype conversion."""

from __future__ import annotations
import numpy
from typing import TYPE_CHECKING
from tensors.backend.storage import Storage
from tensors.backend.numpy.conversion import _storage
from tensors.backend.numpy.conversion import _view

if TYPE_CHECKING:
    from tensors.dtype import DataType
    from tensors.tensor import Tensor


def cast_tensor(value: Tensor, *, dtype: DataType) -> Storage | None:
    """Convert tensor values with Python-compatible scalar conversion."""
    try:
        source = _view(value).reshape(-1)
    except ValueError:
        return None
    if dtype.kind == "integer":
        converter = numpy.frompyfunc(int, 1, 1)
        result = converter(source)
    else:
        result = source.astype(numpy.float64, copy=True)
    return _storage(result, dtype=dtype, output_shape=value.shape)

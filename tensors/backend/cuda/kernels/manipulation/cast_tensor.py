"""CuPy implementation of dtype conversion."""

from __future__ import annotations
import cupy
from typing import TYPE_CHECKING
from tensors.backend.storage import Storage
from tensors.backend.cuda.conversion import _storage
from tensors.backend.cuda.conversion import _view

if TYPE_CHECKING:
    from tensors.dtype import DataType
    from tensors.tensor import Tensor


def cast_tensor(value: Tensor, *, dtype: DataType) -> Storage | None:
    """Convert tensor values with Python-compatible scalar conversion."""
    if dtype.kind == "integer":
        return None
    try:
        source = _view(value).reshape(-1)
    except ValueError:
        return None
    if dtype.kind == "integer":
        converter = cupy.frompyfunc(int, 1, 1)
        result = converter(source)
    else:
        result = source.astype(cupy.float64, copy=True)
    return _storage(result, dtype=dtype, output_shape=value.shape)

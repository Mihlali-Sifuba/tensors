"""CuPy implementation of the rectified linear unit."""

from __future__ import annotations
import cupy
from typing import TYPE_CHECKING
from tensors.backend.storage import Storage
from tensors.backend.cuda.conversion import _errstate
from tensors.backend.cuda.conversion import _storage
from tensors.backend.cuda.conversion import _view

if TYPE_CHECKING:
    from tensors.dtype import DataType
    from tensors.tensor import Tensor


def relu(value: Tensor, *, dtype: DataType) -> Storage | None:
    """Rectify elementwise, leaving NaN in place."""
    if dtype.kind == "integer":
        return None
    try:
        values = _view(value).astype(cupy.float64, copy=False)
    except (TypeError, ValueError):
        return None
    with _errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore"):
        result = cupy.where(cupy.isnan(values), values, cupy.maximum(values, 0.0))
    return _storage(result, dtype=dtype, output_shape=value.shape)

"""CuPy implementation of softplus."""

from __future__ import annotations
import cupy
from typing import TYPE_CHECKING
from tensors.backend.storage import Storage
from tensors.backend.cuda.conversion import _errstate
from tensors.backend.cuda.conversion import _storage
from tensors.backend.cuda.conversion import _working_values

if TYPE_CHECKING:
    from tensors.dtype import DataType
    from tensors.tensor import Tensor


def softplus(value: Tensor, *, dtype: DataType) -> Storage | None:
    """Apply softplus through log1p so a large input does not overflow."""
    if dtype.kind == "integer":
        return None
    try:
        values = _working_values(value)
    except (TypeError, ValueError):
        return None
    with _errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore"):
        result = cupy.log1p(cupy.exp(-cupy.abs(values))) + cupy.maximum(values, 0.0)
    return _storage(result, dtype=dtype, output_shape=value.shape)

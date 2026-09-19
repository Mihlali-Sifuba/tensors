"""CuPy implementation of the logistic function."""

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


def sigmoid(value: Tensor, *, dtype: DataType) -> Storage | None:
    """Apply the logistic function from whichever side avoids overflow."""
    if dtype.kind == "integer":
        return None
    try:
        values = _working_values(value)
    except (TypeError, ValueError):
        return None
    with _errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore"):
        magnitude = cupy.exp(-cupy.abs(values))
        result = cupy.where(
            values >= 0.0, 1.0 / (1.0 + magnitude), magnitude / (1.0 + magnitude)
        )
    return _storage(result, dtype=dtype, output_shape=value.shape)

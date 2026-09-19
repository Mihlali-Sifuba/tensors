"""NumPy implementation of clipping."""

from __future__ import annotations
import numpy
from typing import TYPE_CHECKING
from tensors.backend.storage import Storage
from tensors.backend.numpy.conversion import _storage
from tensors.backend.numpy.conversion import _view

if TYPE_CHECKING:
    from tensors.dtype import DataType
    from tensors.tensor import Tensor


def clip(
    value: Tensor,
    min_value: int | float | None,
    max_value: int | float | None,
    *,
    dtype: DataType,
) -> Storage | None:
    """Clip tensor values to optional scalar bounds."""
    values = _view(value)
    result = numpy.clip(values, min_value, max_value)
    return _storage(result, dtype=dtype, output_shape=value.shape)

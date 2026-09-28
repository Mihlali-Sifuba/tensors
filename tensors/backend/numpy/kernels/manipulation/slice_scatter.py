"""NumPy-native slice scattering."""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

import numpy

from tensors.backend.numpy.conversion import _shape_size, _storage
from tensors.backend.storage import Storage

if TYPE_CHECKING:
    from tensors.dtype import DataType


def slice_scatter(
    value: Any,
    indices: list[int],
    *,
    dtype: DataType,
    output_shape: tuple[int, ...],
) -> Storage | None:
    """Scatter compact source values into a new zero-filled NumPy array."""
    working_dtype = object if dtype.kind == "integer" else numpy.dtype(dtype.name)
    result = numpy.zeros(_shape_size(output_shape), dtype=working_dtype)
    try:
        source = value.reshape(-1).astype(working_dtype, copy=False)
        numpy.add.at(result, indices, source)
    except (TypeError, ValueError):
        return None
    return _storage(result, dtype=dtype, output_shape=output_shape)

"""CUDA-native slice scattering."""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

import cupy

from tensors.backend.cuda.conversion import _shape_size, _storage
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
    """Scatter compact source values into a new zero-filled CUDA array."""
    if dtype.kind == "integer":
        return None
    working_dtype = cupy.dtype(dtype.name)
    result = cupy.zeros(_shape_size(output_shape), dtype=working_dtype)
    try:
        source = value.reshape(-1).astype(working_dtype, copy=False)
        cupy.add.at(result, indices, source)
    except (TypeError, ValueError):
        return None
    return _storage(result, dtype=dtype, output_shape=output_shape)

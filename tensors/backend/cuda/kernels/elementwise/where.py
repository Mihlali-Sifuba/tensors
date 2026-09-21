"""CuPy implementation of elementwise selection."""

from __future__ import annotations
import cupy
from typing import TYPE_CHECKING
from tensors.backend.storage import Storage
from tensors.backend.cuda.conversion import _storage
from tensors.backend.cuda.conversion import tensor_to_logical_array

if TYPE_CHECKING:
    from tensors.dtype import DataType
    from tensors.tensor import Tensor


def where(
    condition: Tensor,
    left: Tensor,
    right: Tensor,
    *,
    dtype: DataType,
    output_shape: tuple[int, ...],
) -> Storage | None:
    """Run broadcasting elementwise selection."""
    try:
        result = cupy.where(
            tensor_to_logical_array(condition) != 0,
            tensor_to_logical_array(left),
            tensor_to_logical_array(right),
        )
    except (TypeError, ValueError):
        return None
    return _storage(result, dtype=dtype, output_shape=output_shape)

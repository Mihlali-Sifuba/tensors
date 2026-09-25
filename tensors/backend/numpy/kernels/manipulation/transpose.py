"""NumPy-native axis permutation."""

from __future__ import annotations

import math
from typing import Any, TYPE_CHECKING

import numpy

from tensors.backend.numpy.storage import NumPyStorage
from tensors.backend.storage import Storage

if TYPE_CHECKING:
    from tensors.dtype import DataType


def transpose(
    value: Any,
    permutation: tuple[int, ...],
    *,
    input_shape: tuple[int, ...],
    dtype: DataType,
    output_shape: tuple[int, ...],
) -> Storage | None:
    """Permute axes into independently owned canonical contiguous storage."""
    try:
        result = numpy.transpose(value.reshape(input_shape), axes=permutation).copy()
    except (TypeError, ValueError):
        return None
    if result.size != math.prod(output_shape):
        raise RuntimeError("transpose kernel returned an unexpected result size")
    return NumPyStorage(result, dtype)

"""NumPy implementation of less."""

from __future__ import annotations

import math
from typing import Any

import numpy

from tensors.backend.numpy.storage import NumPyStorage
from tensors.backend.storage import Storage
from tensors.dtype import uint8


def less(
    left_values: Any, right_values: Any, *, output_shape: tuple[int, ...]
) -> Storage:
    """Evaluate the broadcasting less comparison on native arrays."""
    result = numpy.less(left_values, right_values).astype(numpy.uint8, copy=False)
    storage = NumPyStorage(result, uint8)
    if storage.size != math.prod(output_shape):
        raise RuntimeError("less kernel returned an unexpected result size")
    return storage

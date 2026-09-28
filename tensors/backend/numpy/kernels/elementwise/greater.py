"""NumPy implementation of greater."""

from __future__ import annotations

import math
from typing import Any

import numpy

from tensors.backend.numpy.storage import NumPyStorage
from tensors.backend.storage import Storage
from tensors.dtype import uint8


def greater(
    left_values: Any, right_values: Any, *, output_shape: tuple[int, ...]
) -> Storage:
    """Evaluate the broadcasting greater comparison on native arrays."""
    expected_shape = tuple(output_shape)
    if any(
        tuple(values.shape) != expected_shape for values in (left_values, right_values)
    ):
        raise RuntimeError(
            "greater kernel received operands not prepared for output_shape"
        )
    result = numpy.greater(left_values, right_values).astype(numpy.uint8, copy=False)
    storage = NumPyStorage(result, uint8)
    if storage.size != math.prod(output_shape):
        raise RuntimeError("greater kernel returned an unexpected result size")
    return storage

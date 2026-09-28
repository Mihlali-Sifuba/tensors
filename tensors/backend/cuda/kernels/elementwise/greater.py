"""CUDA implementation of greater."""

from __future__ import annotations

import math
from typing import Any

import cupy

from tensors.backend.cuda.conversion import _widen
from tensors.backend.cuda.storage import CudaStorage
from tensors.backend.storage import Storage
from tensors.dtype import uint8


def greater(
    left_values: Any, right_values: Any, *, output_shape: tuple[int, ...]
) -> Storage:
    """Evaluate the broadcasting greater comparison on device arrays."""
    expected_shape = tuple(output_shape)
    if any(
        tuple(values.shape) != expected_shape for values in (left_values, right_values)
    ):
        raise RuntimeError(
            "greater kernel received operands not prepared for output_shape"
        )
    if left_values.dtype == cupy.float32:
        left_values = _widen(left_values)
    if right_values.dtype == cupy.float32:
        right_values = _widen(right_values)
    result = cupy.greater(left_values, right_values).astype(cupy.uint8, copy=False)
    storage = CudaStorage(result, uint8)
    if storage.size != math.prod(output_shape):
        raise RuntimeError("greater kernel returned an unexpected result size")
    return storage

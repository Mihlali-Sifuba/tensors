"""Dispatch for elementwise operations and their VJPs."""

from __future__ import annotations
from typing import TYPE_CHECKING
from tensors.backend.loading import _backend_kernel
from tensors.backend.policy import (
    _NUMPY_ELEMENTWISE_MIN_SIZE,
    _array_work_is_large_enough,
)
from tensors.backend.storage import Storage

if TYPE_CHECKING:
    from tensors.dtype import DataType
    from tensors.tensor import Tensor


def execute_sqrt(value: Tensor, *, dtype: DataType) -> Storage:
    """Run an accelerated unary transform, or the Python reference."""
    from tensors.backend.python.kernels.elementwise.sqrt import sqrt as reference

    if not _array_work_is_large_enough(value.size, _NUMPY_ELEMENTWISE_MIN_SIZE):
        return reference(value, dtype=dtype)
    unary = _backend_kernel("sqrt")
    result = unary(value, dtype=dtype)
    if result is not None:
        return result
    return reference(value, dtype=dtype)

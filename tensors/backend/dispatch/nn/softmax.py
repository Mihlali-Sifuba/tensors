"""Dispatch for normalization, probability, and loss kernels."""

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


def execute_softmax(value: Tensor, axis: int, *, dtype: DataType) -> Storage:
    """Run a fused softmax-family transform on ordinary finite inputs."""
    from tensors.backend.python.kernels.nn.softmax import softmax as reference

    if not _array_work_is_large_enough(value.size, _NUMPY_ELEMENTWISE_MIN_SIZE):
        return reference(value, axis, dtype=dtype)
    normalization = _backend_kernel("softmax")
    result = normalization(value, axis, dtype=dtype)
    if result is not None:
        return result
    return reference(value, axis, dtype=dtype)

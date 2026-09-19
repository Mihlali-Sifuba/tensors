"""Dispatch for matrix and vector products and their VJPs."""

from __future__ import annotations
from typing import TYPE_CHECKING
from tensors.backend.loading import _backend_kernel
from tensors.backend.policy import _NUMPY_MATMUL_MIN_WORK, _array_work_is_large_enough
from tensors.backend.storage import Storage

if TYPE_CHECKING:
    from tensors.dtype import DataType
    from tensors.tensor import Tensor


def execute_outer(left: Tensor, right: Tensor, *, dtype: DataType) -> Storage:
    """Run a vector outer product with an accelerated backend."""
    from tensors.backend.python.kernels.linalg.outer import outer as reference

    work = left.size * right.size
    if not _array_work_is_large_enough(work, _NUMPY_MATMUL_MIN_WORK):
        return reference(left, right, dtype=dtype)
    outer = _backend_kernel("outer")
    result = outer(left, right, dtype=dtype)
    if result is not None:
        return result
    return reference(left, right, dtype=dtype)

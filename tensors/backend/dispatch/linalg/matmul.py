"""Dispatch for matrix and vector products and their VJPs."""

from __future__ import annotations
from typing import TYPE_CHECKING
from tensors.backend.loading import _backend_kernel
from tensors.backend.policy import (
    _NUMPY_MATMUL_MIN_WORK,
    _array_work_is_large_enough,
    _shape_size,
)
from tensors.backend.storage import Storage

if TYPE_CHECKING:
    from tensors.dtype import DataType
    from tensors.tensor import Tensor


def execute_matmul(
    left: Tensor, right: Tensor, *, dtype: DataType, output_shape: tuple[int, ...]
) -> Storage:
    """Run an accelerated matrix product, or the Python reference."""
    from tensors.backend.python.kernels.linalg.matmul import matmul as reference

    contraction_size = left.shape[-1]
    work = _shape_size(output_shape) * contraction_size
    if not _array_work_is_large_enough(work, _NUMPY_MATMUL_MIN_WORK):
        return reference(left, right, dtype=dtype, output_shape=output_shape)
    matmul = _backend_kernel("matmul")
    result = matmul(left, right, dtype=dtype, output_shape=output_shape)
    if result is not None:
        return result
    return reference(left, right, dtype=dtype, output_shape=output_shape)

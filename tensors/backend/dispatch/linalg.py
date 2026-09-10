"""Dispatch for matrix and vector products and their VJPs."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..loading import _backend_kernel
from ..policy import (
    _NUMPY_MATMUL_MIN_WORK,
    _array_work_is_large_enough,
    _shape_size,
)
from ..storage import Storage

if TYPE_CHECKING:
    from ...dtype import DataType
    from ...tensor import Tensor

def execute_matmul(
    left: Tensor,
    right: Tensor,
    *,
    dtype: DataType,
    output_shape: tuple[int, ...],
) -> Storage | None:
    """Run an accelerated matrix product or request the Python fallback."""
    contraction_size = left.shape[-1]
    work = _shape_size(output_shape) * contraction_size
    if not _array_work_is_large_enough(work, _NUMPY_MATMUL_MIN_WORK):
        return None

    matmul = _backend_kernel("matmul")
    return matmul(left, right, dtype=dtype, output_shape=output_shape)

def execute_matmul_gradient(
    grad: Tensor,
    left: Tensor,
    right: Tensor,
    *,
    needs_input_grad: tuple[bool, ...] = (True, True),
) -> tuple[Storage | None, Storage | None] | None:
    """Run the requested matrix-product VJPs with an accelerated backend.

    ``needs_input_grad`` states which operand gradients the caller wants, so
    the kernel can skip an output the reverse pass will discard. An
    unrequested position comes back as ``None``.
    """
    contraction_size = left.shape[-1]
    work = grad.size * contraction_size
    if not _array_work_is_large_enough(work, _NUMPY_MATMUL_MIN_WORK):
        return None

    matmul_gradient = _backend_kernel("matmul_gradient")
    return matmul_gradient(
        grad,
        left,
        right,
        needs_input_grad=needs_input_grad,
    )

def execute_outer(
    left: Tensor,
    right: Tensor,
    *,
    dtype: DataType,
) -> Storage | None:
    """Run a vector outer product with an accelerated backend."""
    work = left.size * right.size
    if not _array_work_is_large_enough(work, _NUMPY_MATMUL_MIN_WORK):
        return None

    outer = _backend_kernel("outer")
    return outer(left, right, dtype=dtype)

def execute_outer_gradient(
    grad: Tensor,
    left: Tensor,
    right: Tensor,
    *,
    needs_input_grad: tuple[bool, ...] = (True, True),
) -> tuple[Storage | None, Storage | None] | None:
    """Run outer-product VJPs when native sums preserve semantics."""
    if not _array_work_is_large_enough(
        grad.size,
        _NUMPY_MATMUL_MIN_WORK,
    ):
        return None

    outer_gradient = _backend_kernel("outer_gradient")
    return outer_gradient(
        grad,
        left,
        right,
        needs_input_grad=needs_input_grad,
    )

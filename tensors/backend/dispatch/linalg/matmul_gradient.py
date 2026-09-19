"""Dispatch for matrix and vector products and their VJPs."""

from __future__ import annotations
from typing import TYPE_CHECKING
from tensors.backend.loading import _backend_kernel
from tensors.backend.policy import _NUMPY_MATMUL_MIN_WORK, _array_work_is_large_enough
from tensors.backend.storage import Storage

if TYPE_CHECKING:
    from tensors.tensor import Tensor


def execute_matmul_gradient(
    grad: Tensor,
    left: Tensor,
    right: Tensor,
    *,
    needs_input_grad: tuple[bool, ...] = (True, True),
) -> tuple[Storage | None, Storage | None]:
    """Run the requested matrix-product VJPs with an accelerated backend.

    ``needs_input_grad`` states which operand gradients the caller wants, so
    the kernel can skip an output the reverse pass will discard. An
    unrequested position comes back as ``None``.
    """
    from tensors.backend.python.kernels.linalg.matmul_gradient import (
        matmul_gradient as reference,
    )

    contraction_size = left.shape[-1]
    work = grad.size * contraction_size
    if not _array_work_is_large_enough(work, _NUMPY_MATMUL_MIN_WORK):
        return reference(grad, left, right, needs_input_grad=needs_input_grad)
    matmul_gradient = _backend_kernel("matmul_gradient")
    result = matmul_gradient(grad, left, right, needs_input_grad=needs_input_grad)
    if result is not None:
        return result
    return reference(grad, left, right, needs_input_grad=needs_input_grad)

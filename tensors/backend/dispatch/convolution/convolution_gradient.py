"""Dispatch for grouped cross-correlation and its VJPs."""

from __future__ import annotations
from typing import TYPE_CHECKING
from tensors.backend.loading import _backend_kernel
from tensors.backend.policy import _NUMPY_MATMUL_MIN_WORK, _array_work_is_large_enough
from tensors.backend.storage import Storage

if TYPE_CHECKING:
    from tensors.tensor import Tensor


def execute_convolution_gradient(
    grad: Tensor,
    inputs: Tensor,
    kernel: Tensor,
    *,
    stride: tuple[int, ...],
    padding: tuple[int, ...],
    dilation: tuple[int, ...],
    groups: int,
    include_bias: bool,
    needs_input_grad: tuple[bool, ...] = (True, True, True),
) -> tuple[Storage | None, ...]:
    """Run the requested convolution VJPs when native accumulation is safe."""
    from tensors.backend.python.kernels.convolution.convolution_gradient import (
        convolution_gradient as reference,
    )

    patch = kernel.size // max(kernel.shape[0], 1)
    if not _array_work_is_large_enough(grad.size * patch, _NUMPY_MATMUL_MIN_WORK):
        return reference(
            grad,
            inputs,
            kernel,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            include_bias=include_bias,
            needs_input_grad=needs_input_grad,
        )
    convolution_gradient = _backend_kernel("convolution_gradient")
    result = convolution_gradient(
        grad,
        inputs,
        kernel,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        include_bias=include_bias,
        needs_input_grad=needs_input_grad,
    )
    if result is not None:
        return result
    return reference(
        grad,
        inputs,
        kernel,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        include_bias=include_bias,
        needs_input_grad=needs_input_grad,
    )

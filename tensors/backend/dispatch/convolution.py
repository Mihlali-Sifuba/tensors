"""Dispatch for grouped cross-correlation and its VJPs."""

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

def execute_convolution(
    inputs: Tensor,
    kernel: Tensor,
    bias: Tensor | None,
    *,
    dtype: DataType,
    output_shape: tuple[int, ...],
    stride: tuple[int, ...],
    padding: tuple[int, ...],
    dilation: tuple[int, ...],
    groups: int,
) -> Storage | None:
    """Run a grouped cross-correlation with an accelerated backend."""
    patch = kernel.size // max(kernel.shape[0], 1)
    if not _array_work_is_large_enough(
        _shape_size(output_shape) * patch,
        _NUMPY_MATMUL_MIN_WORK,
    ):
        return None

    convolution = _backend_kernel("convolution")
    return convolution(
        inputs,
        kernel,
        bias,
        dtype=dtype,
        output_shape=output_shape,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
    )

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
) -> tuple[Storage | None, ...] | None:
    """Run the requested convolution VJPs when native accumulation is safe."""
    patch = kernel.size // max(kernel.shape[0], 1)
    if not _array_work_is_large_enough(
        grad.size * patch,
        _NUMPY_MATMUL_MIN_WORK,
    ):
        return None

    convolution_gradient = _backend_kernel("convolution_gradient")
    return convolution_gradient(
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

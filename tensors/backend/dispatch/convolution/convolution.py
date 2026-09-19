"""Dispatch for grouped cross-correlation and its VJPs."""

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
) -> Storage:
    """Run a grouped cross-correlation with an accelerated backend."""
    from tensors.backend.python.kernels.convolution.convolution import (
        convolution as reference,
    )

    patch = kernel.size // max(kernel.shape[0], 1)
    if not _array_work_is_large_enough(
        _shape_size(output_shape) * patch, _NUMPY_MATMUL_MIN_WORK
    ):
        return reference(
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
    convolution = _backend_kernel("convolution")
    result = convolution(
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
    if result is not None:
        return result
    return reference(
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

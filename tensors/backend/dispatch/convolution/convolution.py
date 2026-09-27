"""Strict selected-backend dispatch for grouped cross-correlation."""

from __future__ import annotations
from typing import TYPE_CHECKING, Any
from tensors.backend import config
from tensors.backend.config import BackendOperationUnsupportedError
from tensors.backend.loading import load_backend
from tensors.backend.validation import validate_backend_residency

if TYPE_CHECKING:
    from tensors.backend.storage import Storage
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
    """Run grouped cross-correlation on the selected backend without fallback."""
    selected = config.get_backend()
    operands = (inputs, kernel) if bias is None else (inputs, kernel, bias)
    validate_backend_residency(operands, selected)
    backend: Any = load_backend(selected)
    input_buffer = inputs._logical_storage_for(selected).buffer
    kernel_buffer = kernel._logical_storage_for(selected).buffer
    bias_buffer = bias._logical_storage_for(selected).buffer if bias is not None else None
    input_values = input_buffer if selected == "python" else input_buffer.reshape(inputs.shape)
    kernel_values = kernel_buffer if selected == "python" else kernel_buffer.reshape(kernel.shape)
    bias_values = bias_buffer if selected == "python" or bias_buffer is None else bias_buffer.reshape(bias.shape)
    result = backend.convolution(
        input_values,
        kernel_values,
        bias_values,
        tuple(inputs.shape),
        tuple(kernel.shape),
        tuple(bias.shape) if bias is not None else None,
        dtype=dtype,
        output_shape=output_shape,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
    )
    if result is None:
        raise BackendOperationUnsupportedError(
            f"The {selected} backend cannot execute convolution at dtype "
            f"{dtype.name} conformingly. Convolution runs on the selected "
            "backend; select another backend to run it elsewhere."
        )
    validate_backend_residency((result,), selected)
    return result

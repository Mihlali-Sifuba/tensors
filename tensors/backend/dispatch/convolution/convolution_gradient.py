"""Strict selected-backend dispatch for grouped cross-correlation VJPs."""

from __future__ import annotations
from typing import TYPE_CHECKING, Any
from tensors.backend import config
from tensors.backend.config import BackendOperationUnsupportedError
from tensors.backend.loading import load_backend
from tensors.backend.validation import validate_backend_residency

if TYPE_CHECKING:
    from tensors.backend.storage import Storage
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
    """Run requested convolution VJPs on the selected backend without fallback."""
    selected = config.get_backend()
    validate_backend_residency((grad, inputs, kernel), selected)
    backend: Any = load_backend(selected)
    grad_buffer = grad._logical_storage_for(selected).buffer
    input_buffer = inputs._logical_storage_for(selected).buffer
    kernel_buffer = kernel._logical_storage_for(selected).buffer
    grad_values = grad_buffer if selected == "python" else grad_buffer.reshape(grad.shape)
    input_values = input_buffer if selected == "python" else input_buffer.reshape(inputs.shape)
    kernel_values = kernel_buffer if selected == "python" else kernel_buffer.reshape(kernel.shape)
    result = backend.convolution_gradient(
        grad_values,
        input_values,
        kernel_values,
        tuple(grad.shape),
        tuple(inputs.shape),
        tuple(kernel.shape),
        dtype=grad.dtype,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        include_bias=include_bias,
        needs_input_grad=needs_input_grad,
    )
    if result is None:
        raise BackendOperationUnsupportedError(
            f"The {selected} backend cannot execute convolution_gradient at "
            f"dtype {grad.dtype.name} conformingly. The convolution VJP runs "
            "on the selected backend; select another backend to run it elsewhere."
        )
    validate_backend_residency(
        (storage for storage in result if storage is not None), selected
    )
    return result

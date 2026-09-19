"""Dispatch for elementwise operations and their VJPs."""

from __future__ import annotations
from typing import TYPE_CHECKING
from tensors.backend.loading import _backend_kernel
from tensors.backend.policy import (
    _NUMPY_ELEMENTWISE_MIN_SIZE,
    _array_work_is_large_enough,
)
from tensors.backend.storage import Storage

if TYPE_CHECKING:
    from tensors.tensor import Tensor


def execute_clip_gradient(
    grad: Tensor,
    value: Tensor,
    min_value: int | float | None,
    max_value: int | float | None,
) -> Storage:
    """Run the clipping VJP with zero boundary subgradients."""
    from tensors.backend.python.kernels.elementwise.clip_gradient import (
        clip_gradient as reference,
    )

    if not _array_work_is_large_enough(value.size, _NUMPY_ELEMENTWISE_MIN_SIZE):
        return reference(grad, value, min_value, max_value)
    clip_gradient = _backend_kernel("clip_gradient")
    result = clip_gradient(grad, value, min_value, max_value)
    if result is not None:
        return result
    return reference(grad, value, min_value, max_value)

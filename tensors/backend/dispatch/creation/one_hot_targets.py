"""Dispatch for values built from parameters rather than transformed."""

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


def execute_one_hot_targets(logits: Tensor, targets: Tensor, axis: int) -> Storage:
    """Expand class-index targets with the active array backend."""
    from tensors.backend.python.kernels.creation.one_hot_targets import (
        one_hot_targets as reference,
    )

    if not _array_work_is_large_enough(logits.size, _NUMPY_ELEMENTWISE_MIN_SIZE):
        return reference(logits, targets, axis)
    one_hot_targets = _backend_kernel("one_hot_targets")
    result = one_hot_targets(logits, targets, axis)
    if result is not None:
        return result
    return reference(logits, targets, axis)

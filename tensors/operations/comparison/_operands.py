"""Operand preparation shared by the comparison operations.

A comparison is not differentiable and records no graph operation: it reads
both operands as tensors, resolves the broadcast shape, and wraps the mask its
dispatch entry point returns. Only that preparation is shared; each comparison
owns its own module.
"""

from __future__ import annotations

from typing import Any

from tensors.backend.storage import Storage
from tensors.dtype import result_dtype, uint8
from tensors.shape import Shape
from tensors.tensor import Tensor


def _tensor(value: Any, *, reference_dtype: Any = None) -> Tensor:
    from tensors.variable import Variable

    if isinstance(value, Variable):
        return value.data
    if isinstance(value, Tensor):
        return value
    dtype = (
        result_dtype(reference_dtype, value)
        if reference_dtype is not None and isinstance(value, (int, float))
        else None
    )
    return Tensor(value, dtype=dtype)


def comparison_operands(left: Any, right: Any) -> tuple[Tensor, Tensor, Shape]:
    """Return both comparison operands as tensors with their result shape.

    A scalar takes its dtype from the tensor it is compared against, in
    whichever position it appears.
    """
    left_tensor = _tensor(left)
    right_tensor = _tensor(right, reference_dtype=left_tensor.dtype)
    if isinstance(left, (int, float)):
        left_tensor = _tensor(left, reference_dtype=right_tensor.dtype)
    return (
        left_tensor,
        right_tensor,
        left_tensor.shape.broadcast_with(right_tensor.shape),
    )


def comparison_mask(storage: Storage, output_shape: Shape) -> Tensor:
    """Wrap a dispatch result as the ``uint8`` mask a comparison returns."""
    return Tensor._from_owned_storage(storage, dtype=uint8, shape=output_shape)


__all__ = ["comparison_mask", "comparison_operands"]

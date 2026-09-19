"""Elementwise less-than comparison."""

from __future__ import annotations

from tensors._typing import TensorLike
from tensors.backend import execute_less
from tensors.operations.comparison._operands import (
    comparison_mask,
    comparison_operands,
)
from tensors.tensor import Tensor


def less(left: TensorLike, right: TensorLike) -> Tensor:
    """Return the elementwise ``left < right`` mask."""
    left_tensor, right_tensor, shape = comparison_operands(left, right)
    return comparison_mask(
        execute_less(left_tensor, right_tensor, output_shape=shape),
        shape,
    )


__all__ = ["less"]

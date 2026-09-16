"""Elementwise inequality."""

from __future__ import annotations

from tensors._typing import TensorLike
from tensors.backend import execute_not_equal
from tensors.operations.comparison._operands import (
    comparison_mask,
    comparison_operands,
)
from tensors.tensor import Tensor


def not_equal(left: TensorLike, right: TensorLike) -> Tensor:
    """Return an elementwise inequality mask with dtype ``uint8``."""
    left_tensor, right_tensor, shape = comparison_operands(left, right)
    return comparison_mask(
        execute_not_equal(left_tensor, right_tensor, output_shape=shape),
        shape,
    )


__all__ = ["not_equal"]

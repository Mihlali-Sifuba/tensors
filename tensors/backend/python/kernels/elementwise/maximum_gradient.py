"""Python implementation of the maximum VJP."""

from __future__ import annotations

import math
from collections.abc import Iterable

from tensors.backend.python.storage import PythonStorage
from tensors.backend.storage import Storage
from tensors.dtype import DataType


def maximum_gradient(
    grad_values: Iterable[int | float],
    left_values: Iterable[int | float],
    right_values: Iterable[int | float],
    *,
    dtype: DataType,
    output_shape: tuple[int, ...],
    needs_input_grad: tuple[bool, ...] = (True, True),
) -> tuple[Storage | None, Storage | None]:
    """Route gradients to larger values and split exact ties equally."""
    need_left, need_right = needs_input_grad
    left_result = []
    right_result = []
    for upstream, left, right in zip(grad_values, left_values, right_values):
        if math.isnan(left) or math.isnan(right):
            left_weight = right_weight = math.nan
        elif left == right:
            left_weight = right_weight = 0.5
        else:
            left_weight = 1.0 if left > right else 0.0
            right_weight = 1.0 - left_weight
        if need_left:
            left_result.append(upstream * left_weight)
        if need_right:
            right_result.append(upstream * right_weight)
    expected = math.prod(output_shape)
    if (need_left and len(left_result) != expected) or (
        need_right and len(right_result) != expected
    ):
        raise RuntimeError("maximum VJP kernel returned an unexpected result size")
    return (
        PythonStorage.from_values(left_result, dtype) if need_left else None,
        PythonStorage.from_values(right_result, dtype) if need_right else None,
    )

"""Python implementation of the where VJP."""

from __future__ import annotations

import math
from collections.abc import Iterable

from tensors.backend.python.storage import PythonStorage
from tensors.backend.storage import Storage
from tensors.dtype import DataType


def where_gradient(
    grad_values: Iterable[int | float],
    condition_values: Iterable[int | float],
    *,
    dtype: DataType,
    output_shape: tuple[int, ...],
    needs_input_grad: tuple[bool, ...] = (True, True),
) -> tuple[Storage | None, Storage | None]:
    """Route prepared upstream values into the requested data branches."""
    need_left, need_right = needs_input_grad
    pairs = list(zip(grad_values, condition_values))
    left_values = [g if c != 0 else 0.0 for g, c in pairs] if need_left else []
    right_values = [g if c == 0 else 0.0 for g, c in pairs] if need_right else []
    expected = math.prod(output_shape)
    if (need_left and len(left_values) != expected) or (
        need_right and len(right_values) != expected
    ):
        raise RuntimeError("where VJP kernel returned an unexpected result size")
    return (
        PythonStorage.from_values(left_values, dtype) if need_left else None,
        PythonStorage.from_values(right_values, dtype) if need_right else None,
    )

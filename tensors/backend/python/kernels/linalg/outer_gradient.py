"""Reference outer-product VJPs evaluated with Python arithmetic."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from tensors.backend.python.storage import PythonStorage
from tensors.shape import Shape
from tensors.utils.summation import stable_product_sum

if TYPE_CHECKING:
    from tensors.backend.storage import Storage
    from tensors.dtype import DataType


def outer_gradient(
    grad_values: Any,
    left_values: Any,
    right_values: Any,
    *,
    left_shape: tuple[int, ...],
    right_shape: tuple[int, ...],
    dtype: DataType,
    needs_input_grad: tuple[bool, ...] = (True, True),
) -> tuple[Storage | None, Storage | None]:
    """Return the requested outer-product VJPs."""
    left_size = Shape.from_iterable(left_shape).size
    right_size = Shape.from_iterable(right_shape).size
    need_left, need_right = needs_input_grad
    left_gradient = (
        [
            stable_product_sum(
                [
                    (
                        float(grad_values[row * right_size + column]),
                        float(right_values[column]),
                    )
                    for column in range(right_size)
                ]
            )
            for row in range(left_size)
        ]
        if need_left
        else []
    )
    right_gradient = (
        [
            stable_product_sum(
                [
                    (
                        float(grad_values[row * right_size + column]),
                        float(left_values[row]),
                    )
                    for row in range(left_size)
                ]
            )
            for column in range(right_size)
        ]
        if need_right
        else []
    )
    return (
        PythonStorage.from_values(left_gradient, dtype) if need_left else None,
        PythonStorage.from_values(right_gradient, dtype) if need_right else None,
    )

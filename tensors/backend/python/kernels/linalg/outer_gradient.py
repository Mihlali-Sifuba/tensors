"""Reference the vector outer product VJP for the Python backend."""

from __future__ import annotations
from tensors.backend.python.storage import PythonStorage
from tensors.math.sum import _stable_product_sum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tensors.backend.storage import Storage
    from tensors.tensor import Tensor


def outer_gradient(
    grad: Tensor,
    left: Tensor,
    right: Tensor,
    *,
    needs_input_grad: tuple[bool, ...] = (True, True),
) -> tuple[Storage | None, Storage | None] | None:
    """Return the requested outer-product VJPs."""
    need_left, need_right = needs_input_grad
    left_gradient = (
        []
        if not need_left
        else [
            _stable_product_sum(
                [
                    (
                        float(grad._data[row * right.size + column]),
                        float(right._data[column]),
                    )
                    for column in range(right.size)
                ]
            )
            for row in range(left.size)
        ]
    )
    right_gradient = (
        []
        if not need_right
        else [
            _stable_product_sum(
                [
                    (
                        float(grad._data[row * right.size + column]),
                        float(left._data[row]),
                    )
                    for row in range(left.size)
                ]
            )
            for column in range(right.size)
        ]
    )
    return (
        PythonStorage.from_values(left_gradient, grad.dtype) if need_left else None,
        PythonStorage.from_values(right_gradient, grad.dtype) if need_right else None,
    )

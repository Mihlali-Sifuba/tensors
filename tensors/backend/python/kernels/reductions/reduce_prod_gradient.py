"""Reference the product VJP for the Python backend."""

from __future__ import annotations
from tensors.backend.python.storage import PythonStorage
from tensors.math._reduction import reduction_groups
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tensors.backend.storage import Storage
    from tensors.tensor import Tensor


def _product(values: list[int | float]) -> int | float:
    result: int | float = 1
    for value in values:
        result *= value
    return result


def reduce_prod_gradient(
    grad: Tensor, value: Tensor, axes: tuple[int, ...], *, keepdims: bool
) -> Storage | None:
    """Scale each upstream gradient by the product of the others."""
    axis = axes
    _, _, groups = reduction_groups(value, axis, keepdims, scalar_as_vector=True)
    gradients = [0.0] * value.size
    for output_index, group in enumerate(groups):
        for input_index in group:
            other_values = [
                value._data[index] for index in group if index != input_index
            ]
            gradients[input_index] = grad._data[output_index] * _product(other_values)
    return PythonStorage.from_values(gradients, grad.dtype)

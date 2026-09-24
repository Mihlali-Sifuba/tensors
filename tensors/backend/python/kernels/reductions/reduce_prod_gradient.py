"""Reference the product VJP for the Python backend."""

from __future__ import annotations
from tensors.backend.python.storage import PythonStorage
from tensors.dtype import DataType
from tensors.utils.reductions import reduction_groups


def _product(values: list[int | float]) -> int | float:
    result: int | float = 1
    for value in values:
        result *= value
    return result


def reduce_prod_gradient(
    grad_values,
    value_values,
    input_shape: tuple[int, ...],
    axes: tuple[int, ...],
    *,
    keepdims: bool,
    dtype: DataType,
) -> Storage | None:
    """Scale each upstream gradient by the product of the others."""
    axis = axes
    _, _, groups = reduction_groups(input_shape, axis, keepdims, scalar_as_vector=True)
    gradients = [0.0] * len(value_values)
    for output_index, group in enumerate(groups):
        for input_index in group:
            other_values = [
                value_values[index] for index in group if index != input_index
            ]
            gradients[input_index] = grad_values[output_index] * _product(other_values)
    return PythonStorage.from_values(gradients, dtype)

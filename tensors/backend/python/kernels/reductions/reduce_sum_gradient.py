"""Reference the summation VJP for the Python backend."""

from __future__ import annotations
from tensors.backend.python.storage import PythonStorage
from tensors.math._reduction import reduction_groups
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tensors.backend.storage import Storage
    from tensors.tensor import Tensor


def reduce_sum_gradient(
    grad: Tensor, value: Tensor, axes: tuple[int, ...], *, keepdims: bool
) -> Storage | None:
    """Broadcast each upstream gradient back over its group."""
    axis = axes
    a = value
    _, _, groups = reduction_groups(a, axis, keepdims, scalar_as_vector=True)
    result = [0.0] * a.size
    for output_index, group in enumerate(groups):
        for input_index in group:
            result[input_index] = grad._data[output_index]
    return PythonStorage.from_values(result, grad.dtype)

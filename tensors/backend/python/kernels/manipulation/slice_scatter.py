"""Reference slice scattering for the Python backend."""

from __future__ import annotations
from tensors.backend.python.storage import PythonStorage
from tensors.backend.storage import Storage
from tensors.shape import Shape
from tensors.tensor import Tensor


def slice_scatter(
    value: Tensor, indices: list[int], *, output_shape: tuple[int, ...]
) -> Storage | None:
    """Write the source values into the selected positions of a copy."""
    grad = value
    selected = indices
    source_shape = output_shape
    template = Tensor(
        [0.0] * Shape.from_iterable(source_shape).size,
        dtype=grad.dtype,
        shape=source_shape,
    )
    values = [0.0] * template.size
    for logical_linear_index, grad_value in zip(selected, grad._data):
        values[logical_linear_index] += grad_value
    return PythonStorage.from_values(values, grad.dtype)

"""Reference the standard deviation VJP for the Python backend."""

from __future__ import annotations
from tensors.backend.python.storage import PythonStorage
from tensors.tensor import Tensor
from tensors.utils.deviation import scaled_deviations
from tensors.utils.reductions import reduction_groups
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tensors.backend.storage import Storage


def reduce_std_gradient(
    grad: Tensor, value: Tensor, axes: tuple[int, ...], *, keepdims: bool
) -> Storage | None:
    """Differentiate the standard deviation of each group."""
    axis = axes
    _, _, groups = reduction_groups(value.shape, axis, keepdims, scalar_as_vector=True)
    result = [0.0] * value.size
    for output_index, group in enumerate(groups):
        if not group:
            continue
        _, centered, normalized_deviation = scaled_deviations(value._data, group)
        if normalized_deviation == 0:
            continue
        normalizer = len(group) * normalized_deviation
        upstream = grad._data[output_index]
        for input_index, centered_value in zip(group, centered):
            result[input_index] = upstream * (centered_value / normalizer)
    return PythonStorage.from_values(result, grad.dtype)

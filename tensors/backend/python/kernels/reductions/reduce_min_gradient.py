"""Reference the minimum VJP for the Python backend."""

from __future__ import annotations
from tensors.backend.python.storage import PythonStorage
import builtins
import math
from tensors.utils.reductions import reduction_groups
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tensors.backend.storage import Storage
    from tensors.tensor import Tensor


def reduce_min_gradient(
    grad: Tensor, value: Tensor, axes: tuple[int, ...], *, keepdims: bool
) -> Storage | None:
    """Route the upstream gradient to each group's minimum."""
    axis = axes
    _, _, groups = reduction_groups(value.shape, axis, keepdims, scalar_as_vector=True)
    result = [0.0] * value.size
    for output_index, group in enumerate(groups):
        if any(
            (
                isinstance(value._data[index], float) and math.isnan(value._data[index])
                for index in group
            )
        ):
            for input_index in group:
                result[input_index] = math.nan
            continue
        minimum = builtins.min((value._data[index] for index in group))
        selected = [index for index in group if value._data[index] == minimum]
        share = grad._data[output_index] / len(selected)
        for input_index in selected:
            result[input_index] = share
    return PythonStorage.from_values(result, grad.dtype)

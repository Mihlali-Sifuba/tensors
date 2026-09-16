"""Reference the stable log-sum-exp VJP for the Python backend."""

from __future__ import annotations
from tensors.backend.python.storage import PythonStorage
import math
from tensors.math._reduction import reduction_groups
from tensors.math._normalization import shifted_normalization
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tensors.backend.storage import Storage
    from tensors.tensor import Tensor


def logsumexp_gradient(
    grad: Tensor, value: Tensor, axes: tuple[int, ...], *, keepdims: bool
) -> Storage | None:
    """Scale the upstream gradient by the softmax of the group."""
    axis = axes
    a = value
    _, _, groups = reduction_groups(a, axis, keepdims, scalar_as_vector=True)
    values = [0.0] * a.size
    for output_index, group in enumerate(groups):
        if any((math.isnan(float(a._data[index])) for index in group)):
            for input_index in group:
                values[input_index] = math.nan
            continue
        maximum = max((float(a._data[index]) for index in group))
        if maximum == math.inf:
            maxima = [index for index in group if a._data[index] == math.inf]
            weight = 1.0 / len(maxima)
            for input_index in maxima:
                values[input_index] = grad._data[output_index] * weight
            continue
        if maximum == -math.inf:
            raise ValueError(
                "logsumexp gradient is undefined when every reduced value is -inf"
            )
        _, _, probabilities, _ = shifted_normalization(
            [float(a._data[index]) for index in group]
        )
        for input_index, probability in zip(group, probabilities):
            values[input_index] = grad._data[output_index] * probability
    return PythonStorage.from_values(values, grad.dtype)

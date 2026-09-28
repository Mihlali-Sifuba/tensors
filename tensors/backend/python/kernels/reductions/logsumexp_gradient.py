"""Reference the stable log-sum-exp VJP for the Python backend."""

from __future__ import annotations
from tensors.backend.python.storage import PythonStorage
from tensors.dtype import DataType
import math
from tensors.utils.reductions import reduction_groups
from tensors.utils.normalization import shifted_normalization


def logsumexp_gradient(
    grad_values,
    value_values,
    input_shape: tuple[int, ...],
    axes: tuple[int, ...],
    *,
    keepdims: bool,
    dtype: DataType,
) -> Storage | None:
    """Scale the upstream gradient by the softmax of the group."""
    _, _, groups = reduction_groups(input_shape, axes, keepdims, scalar_as_vector=True)
    values = [0.0] * len(value_values)
    for output_index, group in enumerate(groups):
        if any((math.isnan(float(value_values[index])) for index in group)):
            for input_index in group:
                values[input_index] = math.nan
            continue
        maximum = max((float(value_values[index]) for index in group))
        if maximum == math.inf:
            maxima = [index for index in group if value_values[index] == math.inf]
            weight = 1.0 / len(maxima)
            for input_index in maxima:
                values[input_index] = grad_values[output_index] * weight
            continue
        if maximum == -math.inf:
            raise ValueError(
                "logsumexp gradient is undefined when every reduced value is -inf"
            )
        _, _, probabilities, _ = shifted_normalization(
            [float(value_values[index]) for index in group]
        )
        for input_index, probability in zip(group, probabilities):
            values[input_index] = grad_values[output_index] * probability
    return PythonStorage.from_values(values, dtype)

"""Broadcast each upstream derivative, divided by its reduction size."""

from tensors.backend.python.storage import PythonStorage
from tensors.dtype import DataType
from tensors.utils.reductions import reduction_groups


def reduce_mean_gradient(
    grad_values, value_values, input_shape, axes, *, keepdims, dtype
):
    """Spread each upstream gradient evenly over its group."""
    _, _, groups = reduction_groups(input_shape, axes, keepdims, scalar_as_vector=True)
    result = [0.0] * len(value_values)
    for output_index, group in enumerate(groups):
        for input_index in group:
            result[input_index] = float(grad_values[output_index]) / len(group)
    return PythonStorage.from_values(result, dtype)

"""Broadcast each upstream derivative, divided by its reduction size."""

from tensors.backend.python.storage import PythonStorage
from tensors.utils.reductions import reduction_groups


def reduce_mean_gradient(grad, value, axes, *, keepdims):
    """Spread each upstream gradient evenly over its group."""
    _, _, groups = reduction_groups(value.shape, axes, keepdims, scalar_as_vector=True)
    result = [0.0] * value.size
    for output_index, group in enumerate(groups):
        for input_index in group:
            result[input_index] = float(grad._data[output_index]) / len(group)
    return PythonStorage.from_values(result, grad.dtype)

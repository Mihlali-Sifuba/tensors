"""Copy inputs along an existing axis using Python storage."""

import math
from tensors.backend.python.storage import PythonStorage


def concat(values, axis, *, dtype, output_shape):
    """Copy the inputs end to end along an existing axis."""
    trailing = math.prod(output_shape[axis + 1 :])
    groups = math.prod(output_shape[:axis])
    promoted = [
        value if value.dtype == dtype else value.astype(dtype) for value in values
    ]
    result = []
    for group in range(groups):
        for value in promoted:
            count = value.shape[axis] * trailing
            start = group * count
            result.extend(value._data[start : start + count])
    return PythonStorage.from_values(result, dtype)

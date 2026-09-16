"""Copy equal-shaped inputs into a new stacking axis."""

import math
from tensors.backend.python.storage import PythonStorage


def stack(values, axis, *, dtype, output_shape):
    """Copy the inputs into consecutive slots of a new axis."""
    shape = values[0].shape
    before = math.prod(shape[:axis])
    stride = math.prod(shape[axis:])
    result = []
    for group in range(before):
        start = group * stride
        for value in values:
            result.extend(value._data[start : start + stride])
    return PythonStorage.from_values(result, dtype)

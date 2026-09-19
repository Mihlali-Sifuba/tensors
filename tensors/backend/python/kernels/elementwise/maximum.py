"""Python maximum kernel."""

import math
from tensors.backend.python.storage import PythonStorage


def maximum(left, right, *, dtype, output_shape):
    """Select the larger of each broadcast pair, propagating NaN."""
    from tensors.utils.broadcasting import broadcast_to

    def select(x, y):
        if isinstance(x, float) and math.isnan(x):
            return x
        if isinstance(y, float) and math.isnan(y):
            return y
        return x if x >= y else y

    left_values = broadcast_to(left, output_shape)._data
    right_values = broadcast_to(right, output_shape)._data
    values = [select(x, y) for x, y in zip(left_values, right_values)]
    return PythonStorage.from_values(values, dtype)

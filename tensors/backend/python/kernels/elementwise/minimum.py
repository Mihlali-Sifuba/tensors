"""Python minimum kernel."""

import math
from tensors.backend.python.storage import PythonStorage


def minimum(left, right, *, dtype, output_shape):
    """Select the smaller of each broadcast pair, propagating NaN."""
    from tensors.utils.broadcasting import broadcast_binary_values

    def select(x, y):
        if isinstance(x, float) and math.isnan(x):
            return x
        if isinstance(y, float) and math.isnan(y):
            return y
        return x if x <= y else y

    return PythonStorage.from_values(
        broadcast_binary_values(left, right, output_shape, select), dtype
    )

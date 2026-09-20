"""Python greater_equal kernel."""

from tensors.backend.python.storage import PythonStorage
from tensors.dtype import uint8


def greater_equal(left, right, *, output_shape):
    """Return the elementwise ``left >= right`` mask."""
    from tensors.utils.broadcasting import broadcast_to

    left_values = broadcast_to(left, output_shape)._data
    right_values = broadcast_to(right, output_shape)._data
    values = [int(x >= y) for x, y in zip(left_values, right_values)]
    return PythonStorage.from_values(values, uint8)

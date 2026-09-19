"""Python greater kernel."""

from tensors.backend.python.storage import PythonStorage
from tensors.dtype import uint8


def greater(left, right, *, output_shape):
    """Return the elementwise ``left > right`` mask."""
    from tensors.utils.broadcasting import broadcast_binary_values

    values = broadcast_binary_values(left, right, output_shape, lambda x, y: int(x > y))
    return PythonStorage.from_values(values, uint8)

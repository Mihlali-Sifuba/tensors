"""Split upstream values using the broadcast condition."""

from tensors.backend.python.storage import PythonStorage
from tensors.utils.broadcasting import broadcast_to


def where_gradient(grad, condition, *, needs_input_grad=(True, True)):
    """Split the upstream gradient along the condition mask."""
    selected = broadcast_to(condition, grad.shape)
    left = (
        PythonStorage.from_values(
            [g if c != 0 else 0.0 for g, c in zip(grad._data, selected._data)],
            grad.dtype,
        )
        if needs_input_grad[0]
        else None
    )
    right = (
        PythonStorage.from_values(
            [g if c == 0 else 0.0 for g, c in zip(grad._data, selected._data)],
            grad.dtype,
        )
        if needs_input_grad[1]
        else None
    )
    return left, right

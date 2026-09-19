"""Split maximum derivatives, sharing ties equally."""

import math
from tensors.backend.python.storage import PythonStorage
from tensors.utils.broadcasting import broadcast_tensors


def maximum_gradient(grad, left, right, *, needs_input_grad=(True, True)):
    """Route the upstream gradient to the larger operand, sharing ties."""
    left, right = broadcast_tensors(left, right)
    left_values, right_values = [], []
    for upstream, a, b in zip(grad._data, left._data, right._data):
        if math.isnan(a) or math.isnan(b):
            lw = rw = math.nan
        elif a == b:
            lw = rw = 0.5
        else:
            lw = 1.0 if a > b else 0.0
            rw = 1.0 - lw
        if needs_input_grad[0]:
            left_values.append(upstream * lw)
        if needs_input_grad[1]:
            right_values.append(upstream * rw)
    return (
        (
            PythonStorage.from_values(left_values, grad.dtype)
            if needs_input_grad[0]
            else None
        ),
        (
            PythonStorage.from_values(right_values, grad.dtype)
            if needs_input_grad[1]
            else None
        ),
    )

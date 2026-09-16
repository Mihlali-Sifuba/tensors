"""Reference the arctangent VJP for the Python backend."""

from __future__ import annotations
from tensors.backend.python.storage import PythonStorage
from tensors.backend.storage import Storage
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tensors.tensor import Tensor
import math


def _gradient(upstream, value):
    value = float(value)
    if math.isinf(value):
        derivative = 0.0
    elif abs(value) <= 1.0:
        derivative = 1.0 / (1.0 + value * value)
    else:
        reciprocal = 1.0 / value
        square = reciprocal * reciprocal
        derivative = square / (1.0 + square)
    return upstream * derivative


def arctan_gradient(grad: Tensor, value: Tensor) -> Storage:
    """Scale the upstream gradient by ``1 / (1 + x**2)``."""
    evaluate = _gradient
    return PythonStorage.from_values(
        [evaluate(upstream, item) for upstream, item in zip(grad._data, value._data)],
        grad.dtype,
    )

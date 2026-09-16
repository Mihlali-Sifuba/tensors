"""Reference the logistic function VJP for the Python backend."""

from __future__ import annotations
from tensors.backend.python.storage import PythonStorage
from tensors.backend.storage import Storage
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tensors.tensor import Tensor
import math as _math


def _sigmoid_derivative(value: float) -> float:
    """Return the sigmoid derivative without subtracting from rounded one."""
    z = _math.exp(-value) if value >= 0.0 else _math.exp(value)
    denominator = 1.0 + z
    return z / (denominator * denominator)


def sigmoid_gradient(grad: Tensor, value: Tensor) -> Storage:
    """Scale the upstream gradient by ``s * (1 - s)``."""
    evaluate = lambda upstream, value: upstream * _sigmoid_derivative(float(value))
    return PythonStorage.from_values(
        [evaluate(upstream, item) for upstream, item in zip(grad._data, value._data)],
        grad.dtype,
    )

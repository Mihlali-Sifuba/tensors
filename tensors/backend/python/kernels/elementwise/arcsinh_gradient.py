"""Reference the inverse hyperbolic sine VJP for the Python backend."""

from __future__ import annotations
from tensors.backend.python.storage import PythonStorage
from tensors.backend.storage import Storage
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tensors.tensor import Tensor
import math


def _gradient(upstream, value):
    value = float(value)
    magnitude = abs(value)
    if math.isinf(magnitude):
        derivative = 0.0
    elif magnitude <= 1.0:
        derivative = 1.0 / math.sqrt(1.0 + value * value)
    else:
        reciprocal = 1.0 / magnitude
        derivative = reciprocal / math.sqrt(1.0 + reciprocal * reciprocal)
    return upstream * derivative


def arcsinh_gradient(grad: Tensor, value: Tensor) -> Storage:
    """Scale the upstream gradient by ``1 / sqrt(x**2 + 1)``."""
    evaluate = _gradient
    return PythonStorage.from_values(
        [evaluate(upstream, item) for upstream, item in zip(grad._data, value._data)],
        grad.dtype,
    )

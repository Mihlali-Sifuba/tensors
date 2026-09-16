"""Reference the inverse hyperbolic cosine VJP for the Python backend."""

from __future__ import annotations
from tensors.backend.python.storage import PythonStorage
from tensors.backend.storage import Storage
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tensors.tensor import Tensor
import math


def _gradient(upstream, value):
    if value == 1.0:
        raise ValueError("arccosh derivative is undefined at 1")
    value = float(value)
    if math.isinf(value):
        derivative = 0.0
    else:
        derivative = 1.0 / (math.sqrt(value - 1.0) * math.sqrt(value + 1.0))
    return upstream * derivative


def arccosh_gradient(grad: Tensor, value: Tensor) -> Storage:
    """Scale the upstream gradient by ``1 / sqrt(x**2 - 1)``."""
    evaluate = _gradient
    return PythonStorage.from_values(
        [evaluate(upstream, item) for upstream, item in zip(grad._data, value._data)],
        grad.dtype,
    )

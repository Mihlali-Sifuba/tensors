"""Reference the softplus VJP for the Python backend."""

from __future__ import annotations
from tensors.backend.python.storage import PythonStorage
from tensors.backend.storage import Storage
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tensors.tensor import Tensor
import math as _math


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = _math.exp(-value)
        return 1.0 / (1.0 + z)
    z = _math.exp(value)
    return z / (1.0 + z)


def softplus_gradient(grad: Tensor, value: Tensor) -> Storage:
    """Scale the upstream gradient by ``sigmoid(x)``."""
    evaluate = lambda upstream, value: upstream * _sigmoid(float(value))
    return PythonStorage.from_values(
        [evaluate(upstream, item) for upstream, item in zip(grad._data, value._data)],
        grad.dtype,
    )

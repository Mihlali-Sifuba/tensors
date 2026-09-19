"""Reference the square root VJP for the Python backend."""

from __future__ import annotations
from tensors.backend.python.storage import PythonStorage
from tensors.backend.storage import Storage
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tensors.tensor import Tensor
import math as _math


def _sqrt_gradient(upstream, value):
    if value == 0:
        raise ValueError("sqrt derivative is undefined at zero")
    return upstream / (2.0 * _sqrt(value))


def _sqrt(value):
    if value < 0:
        raise ValueError("sqrt is only defined for non-negative values")
    return _math.sqrt(float(value))


def sqrt_gradient(grad: Tensor, value: Tensor) -> Storage:
    """Scale the upstream gradient by ``1 / (2 * sqrt(x))``."""
    evaluate = _sqrt_gradient
    return PythonStorage.from_values(
        [evaluate(upstream, item) for upstream, item in zip(grad._data, value._data)],
        grad.dtype,
    )

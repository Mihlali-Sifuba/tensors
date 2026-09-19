"""Reference the hyperbolic tangent VJP for the Python backend."""

from __future__ import annotations
from tensors.backend.python.storage import PythonStorage
from tensors.backend.storage import Storage
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tensors.tensor import Tensor
import math as _math


def _tanh_derivative(value: float) -> float:
    """Return the tanh derivative without subtracting rounded values."""
    z = _math.exp(-2.0 * abs(value))
    denominator = 1.0 + z
    return 4.0 * z / (denominator * denominator)


def tanh_gradient(grad: Tensor, value: Tensor) -> Storage:
    """Scale the upstream gradient by ``1 - tanh(x)**2``."""
    evaluate = lambda upstream, value: upstream * _tanh_derivative(float(value))
    return PythonStorage.from_values(
        [evaluate(upstream, item) for upstream, item in zip(grad._data, value._data)],
        grad.dtype,
    )

"""Reference the hyperbolic sine VJP for the Python backend."""

from __future__ import annotations
from tensors.backend.python.storage import PythonStorage
from tensors.backend.storage import Storage
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tensors.tensor import Tensor
import math


def _cosh(value):
    try:
        return math.cosh(float(value))
    except OverflowError:
        return math.inf


def sinh_gradient(grad: Tensor, value: Tensor) -> Storage:
    """Scale the upstream gradient by ``cosh(x)``."""
    evaluate = lambda upstream, item: upstream * _cosh(item)
    return PythonStorage.from_values(
        [evaluate(upstream, item) for upstream, item in zip(grad._data, value._data)],
        grad.dtype,
    )

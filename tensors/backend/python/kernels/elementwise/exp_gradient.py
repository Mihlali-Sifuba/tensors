"""Reference the exponential VJP for the Python backend."""

from __future__ import annotations
from tensors.backend.python.storage import PythonStorage
from tensors.backend.storage import Storage
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tensors.tensor import Tensor
import math as _math


def _exp(value):
    try:
        return _math.exp(float(value))
    except OverflowError:
        return _math.inf


def exp_gradient(grad: Tensor, value: Tensor) -> Storage:
    """Scale the upstream gradient by ``exp(x)``."""
    evaluate = lambda upstream, value: upstream * _exp(value)
    return PythonStorage.from_values(
        [evaluate(upstream, item) for upstream, item in zip(grad._data, value._data)],
        grad.dtype,
    )

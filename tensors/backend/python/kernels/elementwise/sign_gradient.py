"""Reference the sign function VJP for the Python backend."""

from __future__ import annotations
from tensors.backend.python.storage import PythonStorage
from tensors.backend.storage import Storage
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tensors.tensor import Tensor
import math


def _gradient(upstream, value):
    if value == 0:
        raise ValueError("sign derivative is undefined at zero")
    if isinstance(value, float) and math.isnan(value):
        return math.nan
    return 0.0


def sign_gradient(grad: Tensor, value: Tensor) -> Storage:
    """Return a zero gradient: the sign function is piecewise constant."""
    evaluate = _gradient
    return PythonStorage.from_values(
        [evaluate(upstream, item) for upstream, item in zip(grad._data, value._data)],
        grad.dtype,
    )

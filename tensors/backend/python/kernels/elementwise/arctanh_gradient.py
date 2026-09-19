"""Reference the inverse hyperbolic tangent VJP for the Python backend."""

from __future__ import annotations
from tensors.backend.python.storage import PythonStorage
from tensors.backend.storage import Storage
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tensors.tensor import Tensor


def arctanh_gradient(grad: Tensor, value: Tensor) -> Storage:
    """Scale the upstream gradient by ``1 / (1 - x**2)``."""
    evaluate = lambda upstream, item: upstream / (1.0 - float(item) * float(item))
    return PythonStorage.from_values(
        [evaluate(upstream, item) for upstream, item in zip(grad._data, value._data)],
        grad.dtype,
    )

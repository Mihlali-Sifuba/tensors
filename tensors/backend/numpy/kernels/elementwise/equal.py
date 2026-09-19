"""NumPy implementation of the equality comparison."""

from __future__ import annotations
import numpy
from typing import TYPE_CHECKING
from tensors.backend.storage import Storage
from tensors.backend.numpy.conversion import _storage
from tensors.backend.numpy.conversion import _view

if TYPE_CHECKING:
    from tensors.tensor import Tensor


def equal(
    left: Tensor, right: Tensor, *, output_shape: tuple[int, ...]
) -> Storage | None:
    """Run a broadcasting elementwise comparison."""
    from tensors.dtype import uint8

    functions = {
        "equal": numpy.equal,
        "not_equal": numpy.not_equal,
        "less": numpy.less,
        "less_equal": numpy.less_equal,
        "greater": numpy.greater,
        "greater_equal": numpy.greater_equal,
    }
    try:
        result = functions["equal"](_view(left), _view(right))
    except (TypeError, ValueError):
        return None
    return _storage(result, dtype=uint8, output_shape=output_shape)

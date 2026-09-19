"""CuPy implementation of the equality comparison."""

from __future__ import annotations
import cupy
from typing import TYPE_CHECKING
from tensors.backend.storage import Storage
from tensors.backend.cuda.conversion import _storage
from tensors.backend.cuda.conversion import _view

if TYPE_CHECKING:
    from tensors.tensor import Tensor


def equal(
    left: Tensor, right: Tensor, *, output_shape: tuple[int, ...]
) -> Storage | None:
    """Run a broadcasting elementwise comparison."""
    from tensors.dtype import uint8

    functions = {
        "equal": cupy.equal,
        "not_equal": cupy.not_equal,
        "less": cupy.less,
        "less_equal": cupy.less_equal,
        "greater": cupy.greater,
        "greater_equal": cupy.greater_equal,
    }
    try:
        result = functions["equal"](_view(left), _view(right))
    except (TypeError, ValueError):
        return None
    return _storage(result, dtype=uint8, output_shape=output_shape)

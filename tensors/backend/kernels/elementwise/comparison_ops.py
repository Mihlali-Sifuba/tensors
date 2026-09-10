"""Elementwise comparison kernels."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ...storage import Storage
from ..core import _numpy, _storage, _view

if TYPE_CHECKING:
    from ....tensor import Tensor
    from ...types import ComparisonOperation

def comparison(
    operation: ComparisonOperation,
    left: Tensor,
    right: Tensor,
    *,
    output_shape: tuple[int, ...],
) -> Storage | None:
    """Run a broadcasting elementwise comparison."""
    from ....dtype import uint8

    numpy = _numpy()
    functions = {
        "equal": numpy.equal,
        "not_equal": numpy.not_equal,
        "less": numpy.less,
        "less_equal": numpy.less_equal,
        "greater": numpy.greater,
        "greater_equal": numpy.greater_equal,
    }
    try:
        result = functions[operation](_view(left, numpy), _view(right, numpy))
    except (TypeError, ValueError):
        return None
    return _storage(
        result,
        dtype=uint8,
        output_shape=output_shape,
        numpy=numpy,
    )

"""Index of the smallest value along a reduction axis."""

from __future__ import annotations

from tensors._typing import TensorLike
from tensors.operations.reductions._arg_extremum import _ArgExtremum, _arg_extremum
from tensors.tensor import Tensor


class ArgMin(_ArgExtremum):
    """Indices of minimum values, selecting the first tie."""


def argmin(
    value: TensorLike, axis: int | None = None, keepdims: bool = False
) -> Tensor:
    """Return first-occurrence indices of minimum values."""
    return _arg_extremum(ArgMin, value, axis, keepdims)


__all__ = ["ArgMin", "argmin"]

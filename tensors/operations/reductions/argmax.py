"""Index of the largest value along a reduction axis."""

from __future__ import annotations

from tensors._typing import TensorLike
from tensors.operations.reductions._arg_extremum import _ArgExtremum, _arg_extremum
from tensors.tensor import Tensor


class ArgMax(_ArgExtremum):
    """Indices of maximum values, selecting the first tie."""

    select_maximum = True


def argmax(
    value: TensorLike, axis: int | None = None, keepdims: bool = False
) -> Tensor:
    """Return first-occurrence indices of maximum values."""
    return _arg_extremum(ArgMax, value, axis, keepdims)


__all__ = ["ArgMax", "argmax"]

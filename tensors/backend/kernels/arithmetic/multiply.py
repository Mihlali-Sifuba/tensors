"""Multiply using an explicitly selected array provider."""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

from ...storage import Storage
from ..core import _errstate, _operand, _storage

if TYPE_CHECKING:
    from ...._typing import Scalar
    from ....dtype import DataType
    from ....tensor import Tensor


def multiply(
    xp: Any,
    left: Tensor | Scalar,
    right: Tensor | Scalar,
    *,
    dtype: DataType,
    output_shape: tuple[int, ...],
) -> Storage | None:
    """Return native storage, or decline when reference semantics require it."""
    try:
        left_array = _operand(left, dtype, xp)
        right_array = _operand(right, dtype, xp)
    except (OverflowError, TypeError, ValueError):
        return None
    with _errstate(
        xp, divide="ignore", over="ignore", under="ignore", invalid="ignore",
    ):
        result = xp.multiply(left_array, right_array)

    return _storage(result, dtype=dtype, output_shape=output_shape, numpy=xp)

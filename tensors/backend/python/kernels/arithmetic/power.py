"""Reference exponentiation for the Python backend."""

from __future__ import annotations
from tensors.backend.python.storage import PythonStorage
from tensors.backend.storage import Storage
from tensors.dtype import DataType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tensors.tensor import Tensor
import math


def _power(base: int | float, exponent: int | float) -> int | float:
    """Calculate a real-valued power with a clear domain error."""
    if isinstance(base, int) and isinstance(exponent, int) and (exponent >= 0):
        return base**exponent
    try:
        value = math.pow(base, exponent)
    except ValueError as exc:
        raise ValueError("power is not defined for these real-valued inputs") from exc
    except OverflowError as exc:
        raise OverflowError("power result is too large to represent") from exc
    return value


def power(
    left: Tensor | int | float,
    right: Tensor | int | float,
    *,
    dtype: DataType,
    output_shape: tuple[int, ...],
) -> Storage:
    """Raise each broadcast pair with Python scalar semantics."""
    from tensors.tensor import Tensor
    from tensors.utils.broadcasting import broadcast_binary_values

    def evaluate(x, y):
        return _power(x, y)

    if isinstance(left, Tensor) and isinstance(right, Tensor):
        values = broadcast_binary_values(left, right, output_shape, evaluate)
    elif isinstance(left, Tensor):
        values = [evaluate(x, right) for x in left._data]
    elif isinstance(right, Tensor):
        values = [evaluate(left, y) for y in right._data]
    else:
        values = [evaluate(left, right)]
    return PythonStorage.from_values(values, dtype)

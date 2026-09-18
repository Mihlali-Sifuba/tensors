"""Reference exponentiation for the Python backend."""

from __future__ import annotations
from tensors.backend.python.storage import PythonStorage
from tensors.backend.storage import Storage
from tensors.dtype import DataType, integer_limits
from tensors.utils.integers import wrap
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


def _integer_power(base: int, exponent: int, dtype: DataType) -> int:
    """Raise one integer pair in the declared width (section 12.4.1).

    ``pow(x, n, m)`` is modular exponentiation by squaring, implemented in C.
    It performs O(log n) multiplications and never materialises ``x ** n``, so
    ``int64: 3 ** 1000000`` costs a few dozen multiplications of 64-bit
    residues rather than building a 1.5-million-bit integer to discard it.

    The residue it returns lies in ``[0, m)``; :func:`wrap` moves it into the
    declared range, which is the identity for an unsigned dtype and the
    two's-complement reading for a signed one.
    """
    lower, upper = integer_limits(dtype)
    modulus = upper - lower + 1
    return wrap(pow(int(base), int(exponent), modulus), dtype)


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

    if dtype.kind == "integer":

        def evaluate(x, y):
            return _integer_power(x, y, dtype)

    else:

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
    return PythonStorage.from_arithmetic(values, dtype)

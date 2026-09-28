"""Reference exponentiation for the Python backend."""

from __future__ import annotations
from tensors.backend.python.storage import PythonStorage
from tensors.backend.storage import Storage
from tensors.dtype import DataType, integer_limits
from tensors.utils.integers import wrap
from collections.abc import Iterable
import math


_INFINITY = float("inf")
_NAN = float("nan")


def _is_odd_integer(value: float) -> bool:
    """Whether a float is an odd integer.

    Every float of magnitude at least ``2 ** 53`` is an even integer, and
    ``value % 2`` reports that correctly, so no separate range test is needed.
    """
    return value.is_integer() and abs(value) % 2.0 == 1.0


def _power(base: int | float, exponent: int | float) -> int | float:
    """IEEE 754-2019 clause 9.2 ``pow`` (sections 12.2 and 12.3).

    Non-trapping: every case in the table of section 12.3.3 is a value, and a
    numerical condition never raises. The order of the tests is the order the
    standard states them, because the first two rows deliberately override NaN
    propagation.

    ``math.pow`` implements the same function but converts three of its
    results into Python exceptions — the invalid case, the divide-by-zero case
    and overflow — so those are decided here and only the ordinary case is
    delegated.
    """
    base = float(base)
    exponent = float(exponent)

    # pow(x, +-0) is 1 for every x, NaN and the infinities included.
    if exponent == 0.0:
        return 1.0
    # pow(1, y) is 1 for every y, NaN included.
    if base == 1.0:
        return 1.0
    # Every other NaN operand propagates.
    if base != base or exponent != exponent:
        return _NAN

    if exponent == _INFINITY or exponent == -_INFINITY:
        magnitude = abs(base)
        if magnitude == 1.0:  # pow(-1, +-inf) is 1
            return 1.0
        return 0.0 if (magnitude < 1.0) == (exponent > 0.0) else _INFINITY

    if base == 0.0:
        odd = _is_odd_integer(exponent)
        if exponent > 0.0:
            return math.copysign(0.0, base) if odd else 0.0
        # A zero denominator: the divide-by-zero signal, not an error.
        return math.copysign(_INFINITY, base) if odd else _INFINITY

    if base == _INFINITY or base == -_INFINITY:
        if base > 0.0:
            return _INFINITY if exponent > 0.0 else 0.0
        odd = _is_odd_integer(exponent)
        if exponent > 0.0:
            return -_INFINITY if odd else _INFINITY
        return -0.0 if odd else 0.0

    # A finite negative base with a non-integral exponent has no real value.
    if base < 0.0 and not exponent.is_integer():
        return _NAN

    try:
        return math.pow(base, exponent)
    except OverflowError:
        # Overflow is a result, not an error; the sign follows the operands.
        negative = base < 0.0 and _is_odd_integer(exponent)
        return -_INFINITY if negative else _INFINITY


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
    left: Iterable[int | float],
    right: Iterable[int | float],
    *,
    dtype: DataType,
    output_shape: tuple[int, ...],
) -> Storage:
    """Raise each prepared pair with Python scalar semantics."""
    if dtype.kind == "integer":

        def evaluate(x, y):
            return _integer_power(x, y, dtype)

    else:

        def evaluate(x, y):
            return _power(x, y)

    values = [evaluate(x, y) for x, y in zip(left, right)]
    return PythonStorage.from_arithmetic(values, dtype)

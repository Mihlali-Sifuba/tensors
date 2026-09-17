"""Fixed-width integer semantics, independent of any backend.

`docs/arithmetic-semantics.md` section 4.2 defines what an integer arithmetic
result is when the exact mathematical result does not fit the declared width.
The rule is stated here once, in terms of plain Python integers, so the
reference backend can apply it and so the meaning does not live inside a
kernel.

Nothing here knows about tensors, storage or backends.
"""

from __future__ import annotations

from tensors.dtype import DataType, integer_limits


def wrap(value: int, dtype: DataType) -> int:
    """Reduce an exact integer result to a dtype's width (section 4.2).

    Signed dtypes use ``((r + 2**(w-1)) mod 2**w) - 2**(w-1)`` and unsigned
    dtypes use ``r mod 2**w``. Both return the unique value in the dtype's
    range congruent to ``r`` modulo ``2**w``, so a result that already fits is
    returned unchanged.

    This is arithmetic, not construction: an out-of-range *operand* or literal
    is an error, and only a computed result wraps.
    """
    lower, upper = integer_limits(dtype)
    modulus = upper - lower + 1
    return (value - lower) % modulus + lower


def wraps_within(value: int, dtype: DataType) -> bool:
    """Whether a value already lies in a dtype's range."""
    lower, upper = integer_limits(dtype)
    return lower <= value <= upper


__all__ = ["wrap", "wraps_within"]

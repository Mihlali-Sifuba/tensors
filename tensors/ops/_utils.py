"""Shared dtype helpers for tensor operations."""

from .. import dtype as _dtype


_FLOAT_CODES = {"f", "d"}
_INTEGER_CODES = {"b", "B", "h", "i", "q"}


def result_dtype(a_dtype, b=None, *, division=False):
    """Choose a predictable result dtype for the supported numeric types."""
    b_dtype = getattr(b, "dtype", None)

    if division:
        if a_dtype.typecode == "d" or getattr(b_dtype, "typecode", None) == "d":
            return _dtype.float64
        if a_dtype.typecode == "f" and (
            b_dtype is None or b_dtype.typecode in _FLOAT_CODES
        ):
            return _dtype.float32
        return _dtype.float64

    if b_dtype is not None:
        if a_dtype == b_dtype:
            return a_dtype
        codes = {a_dtype.typecode, b_dtype.typecode}
        if "d" in codes:
            return _dtype.float64
        if "f" in codes:
            return _dtype.float32
        return a_dtype if a_dtype.size >= b_dtype.size else b_dtype

    if isinstance(b, float) and a_dtype.typecode in _INTEGER_CODES:
        return _dtype.float64
    return a_dtype


def negation_dtype(a_dtype):
    """Unsigned bytes need a signed type to represent negative values."""
    return _dtype.int16 if a_dtype.typecode == "B" else a_dtype

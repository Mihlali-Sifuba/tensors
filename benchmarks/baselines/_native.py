"""The array construction both baselines share.

NumPy and CuPy present the same interface for everything used here, so the
builders are written once against that interface and each adapter supplies
its own module. This is the one place duplication would buy nothing: an
identical second copy could drift, and then two baselines would be holding
different values while claiming to be the same comparison.
"""

from __future__ import annotations

import math
from typing import Any

#: dtype names that hold whole numbers, whose pattern must stay integral.
INTEGER_DTYPES = frozenset({"int64", "int32", "int16", "int8", "uint8"})


def pattern(provider: Any, size: int, dtype_name: str, kind: str) -> Any:
    """Build the value pattern with provider arithmetic rather than a list.

    Building it natively matters: a Python list of a million values would
    cost more to convert than the operation being measured.
    """
    native_dtype = getattr(provider, dtype_name)
    index = provider.arange(size, dtype=provider.int64)
    if dtype_name in INTEGER_DTYPES:
        magnitude = index % 97 + 1
    else:
        magnitude = 1.0 + index % 97 / 97.0
    if kind == "mixed":
        signs = provider.where(index % 2 == 1, 1, -1)
        magnitude = magnitude * signs
    elif kind != "ramp":
        raise ValueError(f"unknown value kind {kind!r}")
    return magnitude.astype(native_dtype)


def build(
    provider: Any,
    shape: tuple[int, ...],
    *,
    dtype_name: str,
    kind: str,
    value: float,
) -> Any:
    """Return a native array matching what the Tensor builder would produce."""
    size = math.prod(shape) if shape else 1
    if kind == "constant":
        return provider.full(shape, value, dtype=getattr(provider, dtype_name))
    return pattern(provider, size, dtype_name, kind).reshape(shape)


__all__ = ["build", "pattern"]

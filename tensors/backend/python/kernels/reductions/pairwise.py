"""Deterministic pairwise floating summation for the Python backend.

The same tree as every other backend (`docs/summation-semantics.md`): a
reduction group in logical order, adjacent pairs added with one rounding to the
declared dtype, an odd final value carried unchanged into the next round,
repeated until one value remains.

Python arithmetic is binary64. For ``float64`` that is the declared dtype. For
``float32`` each addition or product of two binary32 values is computed in
binary64, where it is exact, and then rounded once to binary32 before it takes
part in anything else; that single rounding is the correctly rounded binary32
operation. The whole group is never accumulated in binary64 and narrowed at
the end, which would be a different algorithm.
"""

from __future__ import annotations

import math
from array import array
from collections.abc import Iterable, Sequence

from tensors.dtype import DataType


def round_to_dtype(values: list[float], dtype: DataType) -> list[float]:
    """Round binary64 values to ``dtype``, keeping overflow and subnormals.

    ``array`` stores through a C conversion: round-to-nearest-even, with
    gradual underflow and overflow to a signed infinity.
    """
    if dtype.typecode == "d":
        return values
    return array(dtype.typecode, values).tolist()


def pairwise_float_sum(values: Iterable[float], dtype: DataType) -> float:
    """Sum one group of floats with the package's pairwise tree."""
    level = round_to_dtype([float(value) for value in values], dtype)
    if not level:
        return 0.0
    has_nan = any(math.isnan(value) for value in level)
    has_positive = math.inf in level
    has_negative = -math.inf in level
    while len(level) > 1:
        paired = len(level) - len(level) % 2
        summed = round_to_dtype(
            [level[index] + level[index + 1] for index in range(0, paired, 2)], dtype
        )
        if len(level) % 2:
            summed.append(level[-1])
        level = summed
    result = level[0]
    if has_nan or (has_positive and has_negative):
        return math.nan
    if has_positive:
        return math.inf
    if has_negative:
        return -math.inf
    return 0.0 if result == 0.0 else result


def pairwise_product_sum(
    factors: Sequence[tuple[float, float]], dtype: DataType
) -> float:
    """Form each product in ``dtype``, then sum them with the pairwise tree."""
    products = round_to_dtype(
        [float(left) * float(right) for left, right in factors], dtype
    )
    return pairwise_float_sum(products, dtype)


__all__ = ["pairwise_float_sum", "pairwise_product_sum", "round_to_dtype"]

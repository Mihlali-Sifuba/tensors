"""Accurate summation over plain Python floats.

These primitives are the accumulation rules the reference semantics are
defined by: a sum that survives an overflowing partial total, and a sum of
products whose individual products fall outside float range. They work on
ordinary numbers and sequences, so both the backend kernels that evaluate an
operation and the graph code that differentiates one can use the same
arithmetic without either depending on the other.
"""

from __future__ import annotations

import builtins
import math


def sum_exact_ratios(ratios: list[tuple[int, int]], *, divisor: int = 1) -> float:
    """Convert an exact sum of binary ratios, optionally divided, to a float."""
    denominator = max((item[1] for item in ratios), default=1)
    numerator = builtins.sum(
        (
            item_numerator * (denominator // item_denominator)
            for item_numerator, item_denominator in ratios
        )
    )
    try:
        return numerator / (denominator * divisor)
    except OverflowError:
        return math.inf if numerator > 0 else -math.inf


def stable_float_sum(values: list[float]) -> float:
    """Sum floats accurately even when a temporary partial sum overflows."""
    if any((math.isnan(value) for value in values)):
        return math.nan
    has_positive_infinity = math.inf in values
    has_negative_infinity = -math.inf in values
    if has_positive_infinity and has_negative_infinity:
        return math.nan
    if has_positive_infinity:
        return math.inf
    if has_negative_infinity:
        return -math.inf
    try:
        return math.fsum(values)
    except OverflowError:
        return sum_exact_ratios([value.as_integer_ratio() for value in values])


def stable_product_sum(factors: list[tuple[float, float]]) -> float:
    """Accurately sum products, including products outside float range."""
    if len(factors) == 1:
        left, right = factors[0]
        product = left * right
        if product and math.isfinite(product):
            return product
    products = [left * right for left, right in factors]
    finite_factors = all(
        (math.isfinite(left) and math.isfinite(right) for left, right in factors)
    )
    product_lost_range = any(
        (
            math.isinf(product) or (product == 0.0 and left != 0.0 and (right != 0.0))
            for (left, right), product in zip(factors, products)
        )
    )
    if not finite_factors or not product_lost_range:
        return stable_float_sum(products)
    ratios = []
    for left, right in factors:
        left_numerator, left_denominator = left.as_integer_ratio()
        right_numerator, right_denominator = right.as_integer_ratio()
        ratios.append(
            (left_numerator * right_numerator, left_denominator * right_denominator)
        )
    return sum_exact_ratios(ratios)


def stable_float_mean(values: list[float]) -> float:
    """Return a mean without overflowing its sum or underflowing its terms."""
    if not values:
        return math.nan
    if any((not math.isfinite(value) for value in values)):
        return stable_float_sum(values) / len(values)
    return sum_exact_ratios(
        [value.as_integer_ratio() for value in values], divisor=len(values)
    )


__all__ = [
    "stable_float_mean",
    "stable_float_sum",
    "stable_product_sum",
    "sum_exact_ratios",
]

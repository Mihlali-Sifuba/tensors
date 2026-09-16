"""Reference the division-denominator VJP for the Python backend."""

from __future__ import annotations
from tensors.backend.python.storage import PythonStorage
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tensors.backend.storage import Storage
    from tensors.tensor import Tensor


def _negative_product_over_square(
    left: float, right: float, denominator: float
) -> float:
    """Evaluate ``-left * right / denominator**2`` without range loss."""
    return _product_over_denominator_power(
        [-float(left), float(right)], float(denominator), 2
    )


def _product_over_denominator_power(
    factors: list[float], denominator: float, power: int
) -> float:
    """Evaluate a product divided by a denominator power exactly when finite."""
    import math

    denominator = float(denominator)
    if denominator == 0.0:
        raise ZeroDivisionError("Division by zero")
    if any((value == 0.0 for value in factors)):
        return 0.0
    if all((math.isfinite(value) for value in factors + [denominator])):
        numerator = 1
        divisor = 1
        for factor in factors:
            factor_numerator, factor_denominator = factor.as_integer_ratio()
            numerator *= factor_numerator
            divisor *= factor_denominator
        denominator_numerator, denominator_denominator = denominator.as_integer_ratio()
        numerator *= denominator_denominator**power
        divisor *= denominator_numerator**power
        try:
            return numerator / divisor
        except OverflowError:
            return math.inf if numerator * divisor > 0 else -math.inf
    result = 1.0
    for factor in factors:
        result *= factor
    for _ in range(power):
        result /= denominator
    return result


def division_denominator_gradient(
    grad: Tensor, numerator: Tensor, denominator: Tensor
) -> Storage | None:
    """Scale the upstream gradient by ``-numerator / denominator**2``."""
    values = [
        _negative_product_over_square(upstream, value, divisor)
        for upstream, value, divisor in zip(
            grad._data, numerator._data, denominator._data
        )
    ]
    return PythonStorage.from_values(values, grad.dtype)

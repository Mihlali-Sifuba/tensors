"""Reference the division-denominator VJP for the Python backend."""

from __future__ import annotations
import math
from collections.abc import Iterable
from tensors.backend.python.storage import PythonStorage
from tensors.backend.storage import Storage
from tensors.dtype import DataType
from tensors.utils.broadcasting import broadcast_source_indices

_INFINITY = float("inf")
_NAN = float("nan")


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
    """Evaluate a product divided by a denominator power exactly when finite.

    The exact rational path exists to survive an intermediate that would
    overflow or underflow in floating point. It cannot represent a zero
    denominator, which is not an error but the specified division in
    `docs/arithmetic-semantics.md` section 7.2, so that case is answered
    directly instead.
    """
    denominator = float(denominator)
    factors = [float(value) for value in factors]

    if denominator == 0.0 or not all(math.isfinite(value) for value in factors):
        product = 1.0
        for factor in factors:
            product *= factor
        if denominator != 0.0:
            return product / denominator**power
        if product == 0.0 or product != product:
            return _NAN
        # An even power of a zero is a positive zero whichever sign the
        # denominator carried, so only the product decides the sign.
        sign = 1.0 if power % 2 == 0 else math.copysign(1.0, denominator)
        return math.copysign(_INFINITY, math.copysign(1.0, product) * sign)

    if any(value == 0.0 for value in factors):
        return 0.0
    if math.isfinite(denominator):
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
    grad_values: Iterable[float],
    numerator_values: Iterable[float],
    denominator_values: Iterable[float],
    *,
    dtype: DataType,
    output_shape: tuple[int, ...],
    grad_shape: tuple[int, ...],
    numerator_shape: tuple[int, ...],
    denominator_shape: tuple[int, ...],
) -> Storage:
    """Scale the upstream gradient by ``-numerator / denominator**2``.

    The operands arrive as flat native buffers with their logical shapes.
    Broadcasting is applied as an index mapping, without constructing expanded
    Tensors or moving values through another backend. A zero divisor is
    answered numerically, not refused: section 7.2 specifies the signed
    infinity or the NaN, and the range-safe helper above delivers it.
    """
    grad_indices = broadcast_source_indices(grad_shape, output_shape)
    numerator_indices = broadcast_source_indices(numerator_shape, output_shape)
    denominator_indices = broadcast_source_indices(denominator_shape, output_shape)
    result = [
        _negative_product_over_square(
            grad_values[grad_index],
            numerator_values[numerator_index],
            denominator_values[denominator_index],
        )
        for grad_index, numerator_index, denominator_index in zip(
            grad_indices, numerator_indices, denominator_indices
        )
    ]
    if len(result) != math.prod(output_shape):
        raise RuntimeError(
            "Division denominator VJP kernel returned an unexpected result size"
        )
    return PythonStorage.from_arithmetic(result, dtype)

"""Reference the power-base VJP for the Python backend."""

from __future__ import annotations
from tensors.backend.python.storage import PythonStorage
import math
from tensors.tensor import Tensor
from tensors.dtype import resolve_power
from tensors.backend.python.kernels.arithmetic.power import power, _power
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tensors.backend.storage import Storage


def _power_values(base, exponent):
    # The same resolution the forward operation uses, so a gradient never
    # disagrees with the value it differentiates (section 12.5).
    dtype, exponent = resolve_power(base.dtype, exponent)
    storage = power(base, exponent, dtype=dtype, output_shape=base.shape)
    return Tensor._from_owned_storage(storage, dtype=dtype, shape=base.shape)


def _base_gradient_value(
    upstream: float, base: float, exponent: float, output: float
) -> float:
    """Return ``upstream * exponent * base**(exponent - 1)`` stably."""
    if upstream == 0.0 or exponent == 0.0:
        return 0.0
    if base == 0.0:
        if exponent == 1.0:
            return upstream
        if exponent > 1.0:
            return 0.0
        raise ValueError("power derivative is undefined at a zero base")
    if all((math.isfinite(value) for value in (upstream, base, exponent, output))):
        if output != 0.0:
            return _product_quotient([upstream, exponent, output], [base])
    return _power_product([upstream, exponent], base, exponent - 1.0)


def _power_product(factors: list[float], base: float, exponent: float) -> float:
    """Return ``product(factors) * base**exponent`` stably."""
    if any((value == 0.0 for value in factors)):
        return 0.0
    try:
        power = float(_power(base, exponent))
    except OverflowError:
        power = math.inf
    if power != 0.0 and math.isfinite(power):
        return _product_quotient(factors + [power])
    if (
        base != 0.0
        and all((math.isfinite(value) for value in factors))
        and math.isfinite(base)
        and math.isfinite(exponent)
    ):
        sign = -1.0 if sum((value < 0.0 for value in factors)) % 2 else 1.0
        magnitude_base = abs(base)
        if base < 0.0:
            if not exponent.is_integer():
                raise ValueError("power is not defined for these real-valued inputs")
            if int(exponent) % 2:
                sign = -sign
        logarithm = math.fsum(
            [math.log(abs(value)) for value in factors]
            + [exponent * math.log(magnitude_base)]
        )
        try:
            magnitude = math.exp(logarithm)
        except OverflowError:
            magnitude = math.inf
        return math.copysign(magnitude, sign)
    return _product_quotient(factors + [power])


def _product_quotient(
    numerators: list[float], denominators: list[float] | None = None
) -> float:
    """Evaluate a product quotient without avoidable range loss."""
    denominators = [] if denominators is None else denominators
    if any((math.isnan(value) for value in numerators + denominators)):
        return math.nan
    if any((value == 0.0 for value in denominators)):
        raise ZeroDivisionError("Division by zero")
    if any((value == 0.0 for value in numerators)):
        return 0.0
    if all((math.isfinite(value) for value in numerators + denominators)):
        numerator = 1
        denominator = 1
        for value in numerators:
            value_numerator, value_denominator = value.as_integer_ratio()
            numerator *= value_numerator
            denominator *= value_denominator
        for value in denominators:
            value_numerator, value_denominator = value.as_integer_ratio()
            numerator *= value_denominator
            denominator *= value_numerator
        try:
            return numerator / denominator
        except OverflowError:
            return math.inf if numerator * denominator > 0 else -math.inf
    result = 1.0
    for value in numerators:
        result *= value
    for value in denominators:
        result /= value
    return result


def power_base_gradient(grad: Tensor, base: Tensor, exponent: Tensor) -> Storage | None:
    """Scale the upstream gradient by ``exponent * base ** (exponent - 1)``."""
    from tensors.utils.broadcasting import broadcast_to

    shape = grad.shape.broadcast_with(base.shape).broadcast_with(exponent.shape)
    grad, base, exponent = (
        broadcast_to(value, shape) for value in (grad, base, exponent)
    )
    output = _power_values(base, exponent)
    values = [
        _base_gradient_value(
            float(upstream), float(base_value), float(power), float(result)
        )
        for upstream, base_value, power, result in zip(
            grad._data, base._data, exponent._data, output._data
        )
    ]
    return PythonStorage.from_values(values, grad.dtype)

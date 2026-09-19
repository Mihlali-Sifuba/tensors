"""Reference the power-base VJP for the Python backend."""

from __future__ import annotations
from tensors.backend.python.storage import PythonStorage
import math
from tensors.tensor import Tensor
from tensors.dtype import resolve_power
from tensors.backend.python.kernels.arithmetic.power import power, _power
from tensors.utils.power_gradients import base_derivative
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tensors.backend.storage import Storage


#: The smallest normal magnitude of each floating dtype. Below it a value has
#: lost significand bits to the subnormal range, so the shortcut that divides
#: the forward result by the base would inherit the loss.
_SMALLEST_NORMAL = {"f": 1.1754943508222875e-38, "d": 2.2250738585072014e-308}


def _power_values(base, exponent):
    # The same resolution the forward operation uses, so a gradient never
    # disagrees with the value it differentiates (section 12.5).
    dtype, exponent = resolve_power(base.dtype, exponent)
    storage = power(base, exponent, dtype=dtype, output_shape=base.shape)
    return Tensor._from_owned_storage(storage, dtype=dtype, shape=base.shape)


def _base_gradient_value(
    upstream: float,
    base: float,
    exponent: float,
    output: float,
    normal_minimum: float = 0.0,
) -> float:
    """Return the VJP ``upstream * d(base ** exponent)/d(base)``.

    The derivative follows the region table of section 12.7.2; only the
    ordinary rows evaluate ``exponent * base ** (exponent - 1)``, and those
    are evaluated without avoidable range loss.

    The classified rows are multiplied by the upstream gradient as they
    stand. Nothing here short-circuits a zero upstream against an infinite or
    undefined derivative: ``0 * inf`` and ``0 * NaN`` are NaN, which is the
    honest answer where no derivative exists, and section 12.7 approves no
    convention for that product.
    """
    derivative, _ = base_derivative(base, exponent)
    if derivative is not None:
        return upstream * derivative

    # The ordinary rows. A zero upstream gives exactly zero here because the
    # derivative is finite; the short-circuit is a range guard, not a
    # convention, and it is correct only because of that.
    if upstream == 0.0 or exponent == 0.0:
        return 0.0
    if math.isfinite(base) and base > 0.0 and math.isfinite(exponent):
        # Two forms are available and they fail in opposite directions.
        #
        #   direct:   upstream * exponent * base**(exponent - 1)
        #   quotient: upstream * exponent * base**exponent / base
        #
        # The quotient reuses the forward value, so it survives where
        # base**(exponent - 1) underflows to nothing — at base 1e308 with
        # exponent -1 that power is 1e-616 and the direct form has no digits
        # left. But it inherits whatever the forward value has already lost,
        # so where *that* is subnormal the quotient is the worse of the two:
        # at base 1e-10 with exponent 4 in float32 the forward value is a
        # subnormal and the quotient came out 57 ULP from the correctly
        # rounded derivative.
        #
        # Whichever intermediate is a normal number is therefore preferred,
        # and the logarithmic form below takes over when neither is.
        try:
            shifted = float(_power(base, exponent - 1.0))
        except OverflowError:
            shifted = math.inf
        if math.isfinite(shifted) and abs(shifted) >= normal_minimum:
            return _product_quotient([upstream, exponent, shifted])
        # Second choice, and only second: the forward value may be subnormal
        # and have lost digits, but it still carries more of them than the
        # logarithmic form below, which is accurate to about a part in 1e14.
        if math.isfinite(output) and output != 0.0:
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
                # Section 12.3.3 gives NaN rather than an error, and section
                # 12.7.2 classifies the derivative here as NaN too.
                return math.nan
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
            float(upstream),
            float(base_value),
            float(power),
            float(result),
            _SMALLEST_NORMAL[base.dtype.typecode],
        )
        for upstream, base_value, power, result in zip(
            grad._data, base._data, exponent._data, output._data
        )
    ]
    # Rule G5: the gradient carries the *base's* declared dtype, not the
    # upstream gradient's.
    return PythonStorage.from_values(values, base.dtype)

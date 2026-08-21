"""Element-wise exponentiation and its differentiation rules."""

from __future__ import annotations

import math
from typing import Any, List, Union

from ..dtype import float64, result_dtype
from ..tensor import Tensor
from ..utils.broadcasting import broadcast_shape, broadcast_to, broadcast_tensors
from ._utils import sum_to_shape, sum_to_shape_graph


Scalar = Union[int, float]


def _power_dtype(base: Tensor, exponent: Tensor | Scalar):
    """Choose a dtype that can represent the requested power operation."""
    if isinstance(exponent, Tensor):
        if base.dtype.typecode in {"f", "d"} or exponent.dtype.typecode in {"f", "d"}:
            return result_dtype(base.dtype, exponent)
        if any(value < 0 for value in exponent._data):
            return float64
        return result_dtype(base.dtype, exponent)

    if base.dtype.typecode in {"f", "d"}:
        return result_dtype(base.dtype, exponent)
    if isinstance(exponent, float) or exponent < 0:
        return float64
    return base.dtype


def _power(base: int | float, exponent: int | float) -> int | float:
    """Calculate a real-valued power with a clear domain error."""
    if isinstance(base, int) and isinstance(exponent, int) and exponent >= 0:
        return base ** exponent
    try:
        value = math.pow(base, exponent)
    except ValueError as exc:
        raise ValueError("power is not defined for these real-valued inputs") from exc
    except OverflowError as exc:
        raise OverflowError("power result is too large to represent") from exc
    return value


def _product_quotient(
    numerators: list[float],
    denominators: list[float] | None = None,
) -> float:
    """Evaluate a product quotient without avoidable range loss."""
    denominators = [] if denominators is None else denominators
    if any(math.isnan(value) for value in numerators + denominators):
        return math.nan
    if any(value == 0.0 for value in denominators):
        raise ZeroDivisionError("Division by zero")
    if any(value == 0.0 for value in numerators):
        return 0.0
    if all(math.isfinite(value) for value in numerators + denominators):
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


def _power_product(
    factors: list[float],
    base: float,
    exponent: float,
) -> float:
    """Return ``product(factors) * base**exponent`` stably."""
    if any(value == 0.0 for value in factors):
        return 0.0
    try:
        power = float(_power(base, exponent))
    except OverflowError:
        power = math.inf
    if power != 0.0 and math.isfinite(power):
        return _product_quotient(factors + [power])

    # A rounded power can be zero or infinite even though the surrounding
    # factors bring the complete expression back into the float range. Work
    # in log space for that case instead of committing to the intermediate.
    if (
        base != 0.0
        and all(math.isfinite(value) for value in factors)
        and math.isfinite(base)
        and math.isfinite(exponent)
    ):
        sign = -1.0 if sum(value < 0.0 for value in factors) % 2 else 1.0
        magnitude_base = abs(base)
        if base < 0.0:
            if not exponent.is_integer():
                raise ValueError(
                    "power is not defined for these real-valued inputs"
                )
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


def _base_gradient_value(
    upstream: float,
    base: float,
    exponent: float,
    output: float,
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
    if all(math.isfinite(value) for value in (upstream, base, exponent, output)):
        if output != 0.0:
            return _product_quotient(
                [upstream, exponent, output],
                [base],
            )
    return _power_product([upstream, exponent], base, exponent - 1.0)


def _exponent_gradient_value(
    upstream: float,
    output: float,
    base: float,
    exponent: float,
) -> float:
    """Return ``upstream * output * log(base)`` stably."""
    if upstream == 0.0 or base == 0.0:
        return 0.0
    logarithm = math.log(base)
    if logarithm == 0.0:
        return 0.0
    if output == 0.0 and math.isfinite(base):
        return _power_product(
            [upstream, logarithm],
            base,
            exponent,
        )
    return _product_quotient([upstream, output, logarithm])


class Pow:
    """Element-wise exponentiation with reverse-mode gradient rules."""

    @staticmethod
    def forward(
        base: Tensor,
        exponent: Tensor | Scalar,
        **kwargs: object,
    ) -> Tensor:
        """Raise every element in ``base`` to ``exponent``."""
        dtype = _power_dtype(base, exponent)
        if isinstance(exponent, (int, float)):
            values = [_power(value, exponent) for value in base._data]
            return Tensor(values, dtype=dtype, shape=base.shape)
        if isinstance(exponent, Tensor):
            expanded_base, expanded_exponent = broadcast_tensors(base, exponent)
            values = [
                _power(value, power)
                for value, power in zip(expanded_base._data, expanded_exponent._data)
            ]
            return Tensor(values, dtype=dtype, shape=expanded_base.shape)
        raise TypeError(f"Unsupported exponent type: {type(exponent)}")

    @staticmethod
    def forward_reverse(exponent: Tensor, base: Scalar) -> Tensor:
        """Return ``base`` raised element-wise to ``exponent``."""
        dtype = result_dtype(exponent.dtype, base)
        base_tensor = Tensor([base] * exponent.size, dtype=dtype, shape=exponent.shape)
        return Pow.forward(base_tensor, exponent)

    @staticmethod
    def backward(grad: Tensor, *inputs: Tensor, **kwargs: object) -> List[Tensor]:
        """Return gradients for a power operation's differentiable inputs."""
        if len(inputs) == 2:
            base, exponent = inputs
            expanded_base, expanded_exponent = broadcast_tensors(base, exponent)
            differentiate_base = bool(kwargs.get("differentiate_base", True))
            differentiate_exponent = bool(kwargs.get("differentiate_exponent", True))
            if differentiate_exponent:
                if any(value < 0 for value in expanded_base._data):
                    raise ValueError(
                        "power gradients with respect to a tensor exponent "
                        "require non-negative bases"
                    )
                if any(
                    value == 0 and not power > 0
                    for value, power in zip(
                        expanded_base._data,
                        expanded_exponent._data,
                    )
                ):
                    raise ValueError(
                        "power gradients for a zero base require strictly "
                        "positive exponents"
                    )
            output = Pow.forward(expanded_base, expanded_exponent)
            if differentiate_base:
                base_gradients = [
                    _base_gradient_value(
                        float(upstream),
                        float(value),
                        float(power),
                        float(result),
                    )
                    for upstream, value, power, result in zip(
                        grad._data,
                        expanded_base._data,
                        expanded_exponent._data,
                        output._data,
                    )
                ]
            else:
                base_gradients = [0.0] * grad.size
            base_grad = Tensor(
                base_gradients,
                dtype=grad.dtype,
                shape=grad.shape,
            )
            if differentiate_exponent:
                exponent_grad = Tensor(
                    [
                        _exponent_gradient_value(
                            float(upstream),
                            float(value),
                            float(base_value),
                            float(power),
                        )
                        for upstream, value, base_value, power in zip(
                            grad._data,
                            output._data,
                            expanded_base._data,
                            expanded_exponent._data,
                        )
                    ],
                    dtype=grad.dtype,
                    shape=grad.shape,
                )
            else:
                exponent_grad = Tensor(
                    [0.0] * grad.size,
                    dtype=grad.dtype,
                    shape=grad.shape,
                )
            return [
                sum_to_shape(base_grad, base.shape),
                sum_to_shape(exponent_grad, exponent.shape),
            ]

        exponent = kwargs.get("scalar")
        if not isinstance(exponent, (int, float)):
            raise TypeError("power scalar exponent must be an int or float")

        value = inputs[0]
        if kwargs.get("reverse", False):
            if exponent < 0:
                raise ValueError(
                    "power gradients with respect to an exponent require a positive base"
                )
            if exponent == 0:
                if any(not power > 0 for power in value._data):
                    raise ValueError(
                        "power gradients for a zero base require strictly "
                        "positive exponents"
                    )
                return [
                    Tensor(
                        [0.0] * value.size,
                        dtype=grad.dtype,
                        shape=value.shape,
                    )
                ]
            output = Pow.forward_reverse(value, exponent)
            values = [
                _exponent_gradient_value(
                    float(upstream),
                    float(result),
                    float(exponent),
                    float(power),
                )
                for upstream, result, power in zip(
                    grad._data,
                    output._data,
                    value._data,
                )
            ]
            return [Tensor(values, dtype=grad.dtype, shape=value.shape)]

        if exponent == 0:
            values = [0.0] * value.size
        else:
            output = Pow.forward(value, exponent)
            values = [
                _base_gradient_value(
                    float(upstream),
                    float(base),
                    float(exponent),
                    float(result),
                )
                for upstream, base, result in zip(
                    grad._data,
                    value._data,
                    output._data,
                )
            ]
        return [Tensor(values, dtype=grad.dtype, shape=value.shape)]

    @staticmethod
    def backward_graph(grad, *inputs, **kwargs: object):
        """Build a differentiable VJP for exponentiation."""
        if len(inputs) == 2:
            base, exponent = inputs
            differentiate_base = bool(kwargs.get("differentiate_base", True))
            differentiate_exponent = bool(
                kwargs.get("differentiate_exponent", True)
            )
            expanded_base, expanded_exponent = broadcast_tensors(
                base.data,
                exponent.data,
            )
            if differentiate_exponent:
                if any(value < 0 for value in expanded_base._data):
                    raise ValueError(
                        "power gradients with respect to a tensor exponent "
                        "require non-negative bases"
                    )
                if any(
                    value == 0 and not power > 0
                    for value, power in zip(
                        expanded_base._data,
                        expanded_exponent._data,
                    )
                ):
                    raise ValueError(
                        "power gradients for a zero base require strictly "
                        "positive exponents"
                    )
            base_gradient = (
                _power_base_vjp(
                    grad,
                    base,
                    exponent,
                    differentiate_exponent=differentiate_exponent,
                )
                if differentiate_base
                else _zero_variable(base)
            )
            exponent_gradient = (
                _power_exponent_vjp(
                    grad,
                    base,
                    exponent,
                    differentiate_base=differentiate_base,
                )
                if differentiate_exponent
                else _zero_variable(exponent)
            )
            return [
                sum_to_shape_graph(base_gradient, base.shape),
                sum_to_shape_graph(exponent_gradient, exponent.shape),
            ]

        exponent = kwargs.get("scalar")
        if not isinstance(exponent, (int, float)):
            raise TypeError("power scalar exponent must be an int or float")
        value = inputs[0]
        if kwargs.get("reverse", False):
            if exponent < 0:
                raise ValueError(
                    "power gradients with respect to an exponent require a positive base"
                )
            if exponent == 0:
                if any(not power > 0 for power in value.data._data):
                    raise ValueError(
                        "power gradients for a zero base require strictly "
                        "positive exponents"
                    )
            base = _constant_like(value, float(exponent))
            return [
                _power_exponent_vjp(
                    grad,
                    base,
                    value,
                    differentiate_base=False,
                )
            ]
        power = _constant_like(value, float(exponent))
        return [
            _power_base_vjp(
                grad,
                value,
                power,
                differentiate_exponent=False,
            )
        ]


def _expanded_power_inputs(
    grad: Tensor,
    base: Tensor,
    exponent: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    shape = broadcast_shape(broadcast_shape(grad.shape, base.shape), exponent.shape)
    return (
        broadcast_to(grad, shape),
        broadcast_to(base, shape),
        broadcast_to(exponent, shape),
    )


class PowerBaseGradient:
    """Differentiable range-safe VJP with respect to a power base."""

    @staticmethod
    def forward(
        grad: Tensor,
        base: Tensor,
        exponent: Tensor,
        *,
        differentiate_exponent: bool,
    ) -> Tensor:
        grad, base, exponent = _expanded_power_inputs(grad, base, exponent)
        output = Pow.forward(base, exponent)
        values = [
            _base_gradient_value(
                float(upstream),
                float(base_value),
                float(power),
                float(result),
            )
            for upstream, base_value, power, result in zip(
                grad._data,
                base._data,
                exponent._data,
                output._data,
            )
        ]
        return Tensor(values, dtype=grad.dtype, shape=grad.shape)

    @staticmethod
    def backward(
        outer_grad: Tensor,
        *inputs: Tensor,
        **kwargs: object,
    ) -> List[Tensor]:
        grad, base, exponent = inputs
        differentiate_exponent = bool(
            kwargs.get("differentiate_exponent", True)
        )
        expanded_grad, expanded_base, expanded_exponent = (
            _expanded_power_inputs(grad, base, exponent)
        )
        expanded_outer = broadcast_to(outer_grad, expanded_grad.shape)
        output = Pow.forward(expanded_base, expanded_exponent)
        grad_values = []
        base_values = []
        exponent_values = []
        for outer, upstream, base_value, power, result in zip(
            expanded_outer._data,
            expanded_grad._data,
            expanded_base._data,
            expanded_exponent._data,
            output._data,
        ):
            outer = float(outer)
            upstream = float(upstream)
            base_value = float(base_value)
            power = float(power)
            result = float(result)
            grad_values.append(
                _base_gradient_value(outer, base_value, power, result)
            )
            base_values.append(
                _power_product(
                    [outer, upstream, power, power - 1.0],
                    base_value,
                    power - 2.0,
                )
            )
            if not differentiate_exponent:
                exponent_values.append(0.0)
            elif base_value == 0.0:
                if power > 1.0:
                    exponent_values.append(0.0)
                else:
                    raise ValueError(
                        "higher-order power derivatives are undefined at "
                        "this zero base"
                    )
            else:
                coefficient = math.fsum([
                    1.0,
                    power * math.log(base_value),
                ])
                exponent_values.append(
                    _power_product(
                        [outer, upstream, coefficient],
                        base_value,
                        power - 1.0,
                    )
                )
        shape = expanded_grad.shape
        return [
            sum_to_shape(Tensor(grad_values, dtype=outer_grad.dtype, shape=shape), grad.shape),
            sum_to_shape(Tensor(base_values, dtype=outer_grad.dtype, shape=shape), base.shape),
            sum_to_shape(
                Tensor(exponent_values, dtype=outer_grad.dtype, shape=shape),
                exponent.shape,
            ),
        ]

    @staticmethod
    def backward_graph(outer_grad, *inputs, **kwargs: object):
        from ..math import log

        grad, base, exponent = inputs
        differentiate_exponent = bool(
            kwargs.get("differentiate_exponent", True)
        )
        grad_gradient = _power_base_vjp(
            outer_grad,
            base,
            exponent,
            differentiate_exponent=differentiate_exponent,
        )
        base_gradient = (
            outer_grad
            * grad
            * exponent
            * (exponent - 1.0)
            * (base ** (exponent - 2.0))
        )
        exponent_gradient = (
            outer_grad
            * grad
            * (base ** (exponent - 1.0))
            * (1.0 + exponent * log(base))
            if differentiate_exponent
            else _zero_variable(exponent)
        )
        return [
            sum_to_shape_graph(grad_gradient, grad.shape),
            sum_to_shape_graph(base_gradient, base.shape),
            sum_to_shape_graph(exponent_gradient, exponent.shape),
        ]


class PowerExponentGradient:
    """Differentiable range-safe VJP with respect to a power exponent."""

    @staticmethod
    def forward(grad: Tensor, base: Tensor, exponent: Tensor) -> Tensor:
        grad, base, exponent = _expanded_power_inputs(grad, base, exponent)
        output = Pow.forward(base, exponent)
        values = [
            _exponent_gradient_value(
                float(upstream),
                float(result),
                float(base_value),
                float(power),
            )
            for upstream, result, base_value, power in zip(
                grad._data,
                output._data,
                base._data,
                exponent._data,
            )
        ]
        return Tensor(values, dtype=grad.dtype, shape=grad.shape)

    @staticmethod
    def backward(
        outer_grad: Tensor,
        *inputs: Tensor,
        **kwargs: object,
    ) -> List[Tensor]:
        grad, base, exponent = inputs
        differentiate_base = bool(kwargs.get("differentiate_base", True))
        expanded_grad, expanded_base, expanded_exponent = (
            _expanded_power_inputs(grad, base, exponent)
        )
        expanded_outer = broadcast_to(outer_grad, expanded_grad.shape)
        output = Pow.forward(expanded_base, expanded_exponent)
        grad_values = []
        base_values = []
        exponent_values = []
        for outer, upstream, base_value, power, result in zip(
            expanded_outer._data,
            expanded_grad._data,
            expanded_base._data,
            expanded_exponent._data,
            output._data,
        ):
            outer = float(outer)
            upstream = float(upstream)
            base_value = float(base_value)
            power = float(power)
            result = float(result)
            grad_values.append(
                _exponent_gradient_value(
                    outer,
                    result,
                    base_value,
                    power,
                )
            )
            if base_value == 0.0:
                if not differentiate_base or power > 1.0:
                    base_values.append(0.0)
                    exponent_values.append(0.0)
                    continue
                raise ValueError(
                    "higher-order power derivatives are undefined at this "
                    "zero base"
                )
            logarithm = math.log(base_value)
            coefficient = math.fsum([1.0, power * logarithm])
            base_values.append(
                _power_product(
                    [outer, upstream, coefficient],
                    base_value,
                    power - 1.0,
                )
            )
            exponent_values.append(
                _power_product(
                    [outer, upstream, logarithm, logarithm],
                    base_value,
                    power,
                )
            )
        shape = expanded_grad.shape
        return [
            sum_to_shape(Tensor(grad_values, dtype=outer_grad.dtype, shape=shape), grad.shape),
            sum_to_shape(Tensor(base_values, dtype=outer_grad.dtype, shape=shape), base.shape),
            sum_to_shape(
                Tensor(exponent_values, dtype=outer_grad.dtype, shape=shape),
                exponent.shape,
            ),
        ]

    @staticmethod
    def backward_graph(outer_grad, *inputs, **kwargs: object):
        from ..math import log

        grad, base, exponent = inputs
        differentiate_base = bool(kwargs.get("differentiate_base", True))
        logarithm = log(base)
        output = base ** exponent
        grad_gradient = _power_exponent_vjp(
            outer_grad,
            base,
            exponent,
            differentiate_base=differentiate_base,
        )
        base_gradient = (
            outer_grad
            * grad
            * (base ** (exponent - 1.0))
            * (1.0 + exponent * logarithm)
            if differentiate_base
            else _zero_variable(base)
        )
        exponent_gradient = outer_grad * grad * output * (logarithm ** 2.0)
        return [
            sum_to_shape_graph(grad_gradient, grad.shape),
            sum_to_shape_graph(base_gradient, base.shape),
            sum_to_shape_graph(exponent_gradient, exponent.shape),
        ]


def _zero_variable(reference):
    from ..variable import Variable

    return Variable(
        Tensor(
            [0.0] * reference.size,
            dtype=reference.dtype,
            shape=reference.shape,
        ),
        requires_grad=False,
    )


def _constant_like(reference, value: float):
    from ..variable import Variable

    return Variable(
        Tensor(
            [value] * reference.size,
            dtype=reference.dtype,
            shape=reference.shape,
        ),
        requires_grad=False,
    )


def _power_base_vjp(
    grad,
    base,
    exponent,
    *,
    differentiate_exponent: bool,
):
    from ..variable import Variable

    return Variable._from_operation(
        PowerBaseGradient.forward(
            grad.data,
            base.data,
            exponent.data,
            differentiate_exponent=differentiate_exponent,
        ),
        "power_base_gradient",
        PowerBaseGradient,
        [grad, base, exponent],
        differentiate_exponent=differentiate_exponent,
    )


def _power_exponent_vjp(
    grad,
    base,
    exponent,
    *,
    differentiate_base: bool,
):
    from ..variable import Variable

    return Variable._from_operation(
        PowerExponentGradient.forward(grad.data, base.data, exponent.data),
        "power_exponent_gradient",
        PowerExponentGradient,
        [grad, base, exponent],
        differentiate_base=differentiate_base,
    )


def pow(base: Any, exponent: Any) -> Any:
    """Return the element-wise power of two Tensors, Variables, or scalars."""
    return base ** exponent


__all__ = ["Pow", "pow"]

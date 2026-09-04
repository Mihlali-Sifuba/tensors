"""Element-wise exponentiation and its differentiation rules."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, List, Optional, Union, overload

from ..backend import (
    execute_binary,
    execute_power_base_gradient,
    execute_power_exponent_gradient,
)
from .._typing import TensorData, TensorLike, TensorResult
from ..dtype import float64, result_dtype
from .operation import Operation
from ..shape import Shape
from ..tensor import Tensor
from ..utils.broadcasting import broadcast_to, broadcast_tensors

if TYPE_CHECKING:
    from ..variable import Variable
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


class Pow(Operation):
    """Element-wise exponentiation with reverse-mode gradient rules."""

    __slots__ = ()
    name = "pow"

    def forward(
        self,
        base: Tensor,
        exponent: Tensor | Scalar,
    ) -> Tensor:
        """Raise every element in ``base`` to ``exponent``."""
        if not isinstance(exponent, (int, float, Tensor)):
            raise TypeError(f"Unsupported exponent type: {type(exponent)}")
        dtype = _power_dtype(base, exponent)
        output_shape = (
            base.shape.broadcast_with(exponent.shape)
            if isinstance(exponent, Tensor)
            else base.shape
        )
        accelerated = execute_binary(
            "power",
            base,
            exponent,
            dtype=dtype,
            output_shape=output_shape,
        )
        if accelerated is not None:
            return Tensor._from_owned_storage(accelerated, dtype=dtype, shape=output_shape)
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

    def backward(
        self,
        grad: Tensor,
        *inputs: Tensor,
        needs_input_grad: tuple[bool, ...],
    ) -> List[Optional[Tensor]]:
        """Return the requested VJPs for a power invocation.

        Only a requested derivative is calculated, and a derivative's domain
        check runs only when the derivative it guards was requested.
        """
        base, exponent = inputs
        need_base, need_exponent = needs_input_grad

        # Broadcasting and the reference power evaluation are only needed by
        # the domain checks and the reference gradient paths. An accelerated
        # backend would otherwise pay for a host materialization it discards.
        expanded: list[Tensor] = []

        def expanded_operands() -> tuple[Tensor, Tensor]:
            if not expanded:
                expanded.extend(broadcast_tensors(base, exponent))
            return expanded[0], expanded[1]

        reference_output: list[Tensor] = []

        def output_values() -> Tensor:
            if not reference_output:
                reference_output.append(self.forward(*expanded_operands()))
            return reference_output[0]

        if need_exponent:
            expanded_base, expanded_exponent = expanded_operands()
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
        if need_base:
            accelerated_base = execute_power_base_gradient(
                grad,
                base,
                exponent,
            )
            if accelerated_base is not None:
                base_grad = Tensor._from_owned_storage(
                    accelerated_base,
                    dtype=grad.dtype,
                    shape=grad.shape,
                )
            else:
                reference_base, reference_exponent = expanded_operands()
                base_grad = Tensor(
                    [
                        _base_gradient_value(
                            float(upstream),
                            float(value),
                            float(power),
                            float(result),
                        )
                        for upstream, value, power, result in zip(
                            grad._data,
                            reference_base._data,
                            reference_exponent._data,
                            output_values()._data,
                        )
                    ],
                    dtype=grad.dtype,
                    shape=grad.shape,
                )
        else:
            # A frozen operand contributes zeros, so build them at the
            # operand's own shape instead of reducing a broadcast zero.
            base_grad = None
        if need_exponent:
            accelerated_exponent = execute_power_exponent_gradient(
                grad,
                base,
                exponent,
            )
            if accelerated_exponent is not None:
                exponent_grad = Tensor._from_owned_storage(
                    accelerated_exponent,
                    dtype=grad.dtype,
                    shape=grad.shape,
                )
            else:
                reference_base, reference_exponent = expanded_operands()
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
                            output_values()._data,
                            reference_base._data,
                            reference_exponent._data,
                        )
                    ],
                    dtype=grad.dtype,
                    shape=grad.shape,
                )
        else:
            exponent_grad = None
        return [
            sum_to_shape(base_grad, base.shape)
            if base_grad is not None
            else None,
            sum_to_shape(exponent_grad, exponent.shape)
            if exponent_grad is not None
            else None,
        ]

    def backward_graph(
        self,
        grad,
        *inputs,
        needs_input_grad: tuple[bool, ...],
    ):
        """Build the requested differentiable VJPs for exponentiation."""
        base, exponent = inputs
        need_base, need_exponent = needs_input_grad
        if need_exponent:
            # Only the exponent's domain checks need the broadcast operands.
            expanded_base, expanded_exponent = broadcast_tensors(
                base.data,
                exponent.data,
            )
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
        return [
            sum_to_shape_graph(
                _power_base_vjp(grad, base, exponent),
                base.shape,
            )
            if need_base
            else None,
            sum_to_shape_graph(
                _power_exponent_vjp(grad, base, exponent),
                exponent.shape,
            )
            if need_exponent
            else None,
        ]



def _reduced(
    values: list[float],
    reference: Tensor,
    shape: Shape,
    target: Tensor,
) -> Tensor:
    """Reduce accumulated per-element VJP values back to an operand shape."""
    return sum_to_shape(
        Tensor(values, dtype=reference.dtype, shape=shape),
        target.shape,
    )


def _expanded_power_inputs(
    grad: Tensor,
    base: Tensor,
    exponent: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    shape = grad.shape.broadcast_with(base.shape).broadcast_with(exponent.shape)
    return (
        broadcast_to(grad, shape),
        broadcast_to(base, shape),
        broadcast_to(exponent, shape),
    )


class PowerBaseGradient(Operation):
    """Differentiable range-safe VJP with respect to a power base."""

    __slots__ = ()
    name = "power_base_gradient"

    def forward(
        self,
        grad: Tensor,
        base: Tensor,
        exponent: Tensor,
    ) -> Tensor:
        grad, base, exponent = _expanded_power_inputs(grad, base, exponent)
        accelerated = execute_power_base_gradient(grad, base, exponent)
        if accelerated is not None:
            return Tensor._from_owned_storage(
                accelerated,
                dtype=grad.dtype,
                shape=grad.shape,
            )
        output = _power_values(base, exponent)
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

    def backward(
        self,
        outer_grad: Tensor,
        *inputs: Tensor,
        needs_input_grad: tuple[bool, ...],
    ) -> List[Optional[Tensor]]:
        grad, base, exponent = inputs
        need_grad, need_base, need_exponent = needs_input_grad
        expanded_grad, expanded_base, expanded_exponent = (
            _expanded_power_inputs(grad, base, exponent)
        )
        expanded_outer = broadcast_to(outer_grad, expanded_grad.shape)
        output = _power_values(expanded_base, expanded_exponent)
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
            if need_grad:
                grad_values.append(
                    _base_gradient_value(outer, base_value, power, result)
                )
            if need_base:
                base_values.append(
                    _power_product(
                        [outer, upstream, power, power - 1.0],
                        base_value,
                        power - 2.0,
                    )
                )
            if not need_exponent:
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
            _reduced(grad_values, outer_grad, shape, grad) if need_grad else None,
            _reduced(base_values, outer_grad, shape, base) if need_base else None,
            _reduced(exponent_values, outer_grad, shape, exponent)
            if need_exponent
            else None,
        ]

    def backward_graph(
        self,
        outer_grad,
        *inputs,
        needs_input_grad: tuple[bool, ...],
    ):
        from ..math import log

        grad, base, exponent = inputs
        need_grad, need_base, need_exponent = needs_input_grad
        return [
            sum_to_shape_graph(
                _power_base_vjp(outer_grad, base, exponent),
                grad.shape,
            )
            if need_grad
            else None,
            sum_to_shape_graph(
                outer_grad
                * grad
                * exponent
                * (exponent - 1.0)
                * (base ** (exponent - 2.0)),
                base.shape,
            )
            if need_base
            else None,
            sum_to_shape_graph(
                outer_grad
                * grad
                * (base ** (exponent - 1.0))
                * (1.0 + exponent * log(base)),
                exponent.shape,
            )
            if need_exponent
            else None,
        ]


class PowerExponentGradient(Operation):
    """Differentiable range-safe VJP with respect to a power exponent."""

    __slots__ = ()
    name = "power_exponent_gradient"

    def forward(self, grad: Tensor, base: Tensor, exponent: Tensor) -> Tensor:
        grad, base, exponent = _expanded_power_inputs(grad, base, exponent)
        accelerated = execute_power_exponent_gradient(
            grad,
            base,
            exponent,
        )
        if accelerated is not None:
            return Tensor._from_owned_storage(
                accelerated,
                dtype=grad.dtype,
                shape=grad.shape,
            )
        output = _power_values(base, exponent)
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

    def backward(
        self,
        outer_grad: Tensor,
        *inputs: Tensor,
        needs_input_grad: tuple[bool, ...],
    ) -> List[Optional[Tensor]]:
        grad, base, exponent = inputs
        need_grad, need_base, need_exponent = needs_input_grad
        expanded_grad, expanded_base, expanded_exponent = (
            _expanded_power_inputs(grad, base, exponent)
        )
        expanded_outer = broadcast_to(outer_grad, expanded_grad.shape)
        output = _power_values(expanded_base, expanded_exponent)
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
            if need_grad:
                grad_values.append(
                    _exponent_gradient_value(
                        outer,
                        result,
                        base_value,
                        power,
                    )
                )
            if base_value == 0.0:
                if not need_base or power > 1.0:
                    base_values.append(0.0)
                    exponent_values.append(0.0)
                    continue
                raise ValueError(
                    "higher-order power derivatives are undefined at this "
                    "zero base"
                )
            logarithm = math.log(base_value)
            if need_base:
                coefficient = math.fsum([1.0, power * logarithm])
                base_values.append(
                    _power_product(
                        [outer, upstream, coefficient],
                        base_value,
                        power - 1.0,
                    )
                )
            if need_exponent:
                exponent_values.append(
                    _power_product(
                        [outer, upstream, logarithm, logarithm],
                        base_value,
                        power,
                    )
                )
        shape = expanded_grad.shape
        return [
            _reduced(grad_values, outer_grad, shape, grad) if need_grad else None,
            _reduced(base_values, outer_grad, shape, base)
            if need_base
            else None,
            _reduced(exponent_values, outer_grad, shape, exponent)
            if need_exponent
            else None,
        ]

    def backward_graph(
        self,
        outer_grad,
        *inputs,
        needs_input_grad: tuple[bool, ...],
    ):
        from ..math import log

        grad, base, exponent = inputs
        need_grad, need_base, need_exponent = needs_input_grad
        logarithm = log(base)
        return [
            sum_to_shape_graph(
                _power_exponent_vjp(outer_grad, base, exponent),
                grad.shape,
            )
            if need_grad
            else None,
            sum_to_shape_graph(
                outer_grad
                * grad
                * (base ** (exponent - 1.0))
                * (1.0 + exponent * logarithm),
                base.shape,
            )
            if need_base
            else None,
            sum_to_shape_graph(
                outer_grad * grad * (base ** exponent) * (logarithm ** 2.0),
                exponent.shape,
            )
            if need_exponent
            else None,
        ]


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


def _power_base_vjp(grad, base, exponent):
    from ..variable import Variable

    operation = PowerBaseGradient()
    return Variable._from_operation(
        operation.forward(grad.data, base.data, exponent.data),
        operation,
        (grad, base, exponent),
    )


def _power_exponent_vjp(grad, base, exponent):
    from ..variable import Variable

    operation = PowerExponentGradient()
    return Variable._from_operation(
        operation.forward(grad.data, base.data, exponent.data),
        operation,
        (grad, base, exponent),
    )


@overload
def pow(base: Variable, exponent: TensorLike) -> Variable: ...


@overload
def pow(base: TensorLike, exponent: Variable) -> Variable: ...


@overload
def pow(base: TensorData, exponent: TensorData) -> Tensor: ...


def pow(base: TensorLike, exponent: TensorLike) -> TensorResult:
    """Return the element-wise power of two Tensors, Variables, or scalars."""
    return base ** exponent


_power_values = Pow().forward
power = _power_values


def power_scalar_base(base: Scalar, exponent: Tensor) -> Tensor:
    """Return ``base`` raised element-wise to ``exponent`` for a scalar base."""
    dtype = result_dtype(exponent.dtype, base)
    accelerated = execute_binary(
        "power",
        base,
        exponent,
        dtype=dtype,
        output_shape=exponent.shape,
    )
    if accelerated is not None:
        return Tensor._from_owned_storage(
            accelerated,
            dtype=dtype,
            shape=exponent.shape,
        )
    base_tensor = Tensor(
        [base] * exponent.size,
        dtype=dtype,
        shape=exponent.shape,
    )
    return _power_values(base_tensor, exponent)


__all__ = ["Pow", "pow", "power", "power_scalar_base"]

"""Element-wise exponentiation and its differentiation rules."""

from __future__ import annotations
import math
from typing import TYPE_CHECKING, List, Optional, Union, overload
from tensors.backend import (
    execute_power,
    execute_power_base_gradient,
    execute_power_exponent_gradient,
)
from tensors._typing import TensorData, TensorLike, TensorResult
from tensors.dtype import resolve_power, resolve_power_scalar_base
from tensors.operations.base import Operation
from tensors.shape import Shape
from tensors.tensor import Tensor
from tensors.utils.broadcasting import broadcast_to, broadcast_tensors
from tensors.utils.power_gradients import base_derivative, exponent_derivative

if TYPE_CHECKING:
    from tensors.variable import Variable
from tensors.operations._gradient_shaping import sum_to_shape, sum_to_shape_graph

Scalar = Union[int, float]

#: The smallest normal binary32 magnitude. The gradient shortcut that
#: divides the forward value by the base is exact enough only above it;
#: the narrower dtype's threshold is used so neither dtype is served by a
#: value that has already lost significand bits.
_SMALLEST_NORMAL = 1.1754943508222875e-38


def _has_negative_exponent(exponent: Tensor) -> bool:
    """Whether an integer exponent tensor holds a negative element.

    Integer exponentiation has no fractional value to deliver, so section
    12.4.2 requires this test, and it runs only when the result dtype is an
    integer dtype: a floating power delivers the IEEE result and must never
    read its exponent.

    The test asks the native buffer rather than materialising the tensor, so a
    device exponent costs one reduction and one scalar transfer, not a copy of
    the tensor. A view is resolved to its logical values first, so elements the
    exponent does not address cannot make it raise.
    """
    storage = exponent._logical_storage_for(exponent.backend_storage.kind)
    buffer = storage.buffer
    if getattr(buffer, "any", None) is None:
        return any(value < 0 for value in buffer)
    return bool((buffer < 0).any())


def _reject_negative_exponent(dtype, exponent) -> None:
    """Apply section 12.4.2 once the result dtype is settled.

    The dtype is decided first, from declarations alone (section 12.5), and
    only then is the exponent's domain examined. The two stages never
    interact: a negative exponent never promotes the result to a floating
    dtype, it refuses the operation.
    """
    if dtype.kind != "integer":
        return
    negative = (
        _has_negative_exponent(exponent)
        if isinstance(exponent, Tensor)
        else exponent < 0
    )
    if negative:
        raise ValueError(
            "integer exponentiation requires a non-negative exponent; "
            + dtype.name
            + " cannot represent a reciprocal. Cast the base to a floating "
            "dtype, for example base.astype(ts.float64) ** exponent"
        )


def _power(base: int | float, exponent: int | float) -> int | float:
    """Calculate a real-valued power with a clear domain error."""
    if isinstance(base, int) and isinstance(exponent, int) and (exponent >= 0):
        return base**exponent
    try:
        value = math.pow(base, exponent)
    except ValueError as exc:
        raise ValueError("power is not defined for these real-valued inputs") from exc
    except OverflowError as exc:
        raise OverflowError("power result is too large to represent") from exc
    return value


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
                # Section 12.3.3 gives NaN rather than an error here.
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


def _base_gradient_value(
    upstream: float,
    base: float,
    exponent: float,
    output: float,
    normal_minimum: float = 0.0,
) -> float:
    """Return the VJP with respect to the base, by the section 12.7.2 table."""
    derivative, _ = base_derivative(base, exponent)
    if derivative is not None:
        return upstream * derivative
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


def _exponent_gradient_value(
    upstream: float, output: float, base: float, exponent: float
) -> float:
    """Return the VJP with respect to the exponent, by the same table."""
    derivative, _ = exponent_derivative(base, exponent)
    if derivative is not None:
        return upstream * derivative
    if upstream == 0.0:
        return 0.0
    logarithm = math.log(base)
    if logarithm == 0.0:
        return 0.0
    if output == 0.0 and math.isfinite(base):
        return _power_product([upstream, logarithm], base, exponent)
    return _product_quotient([upstream, output, logarithm])


class Pow(Operation):
    """Element-wise exponentiation with reverse-mode gradient rules."""

    __slots__ = ()
    name = "pow"

    def forward(self, base: Tensor, exponent: Tensor | Scalar) -> Tensor:
        """Raise every element in ``base`` to ``exponent``."""
        if not isinstance(exponent, (int, float, Tensor)):
            raise TypeError(f"Unsupported exponent type: {type(exponent)}")
        # Promotion for a typed exponent, conversion for a scalar; section
        # 12.5. No element value is read.
        dtype, exponent = resolve_power(base.dtype, exponent)
        _reject_negative_exponent(dtype, exponent)
        output_shape = (
            base.shape.broadcast_with(exponent.shape)
            if isinstance(exponent, Tensor)
            else base.shape
        )
        accelerated = execute_power(
            base, exponent, dtype=dtype, output_shape=output_shape
        )
        return Tensor._from_owned_storage(accelerated, dtype=dtype, shape=output_shape)

    def backward(
        self, grad: Tensor, *inputs: Tensor, needs_input_grad: tuple[bool, ...]
    ) -> List[Optional[Tensor]]:
        """Return the requested VJPs for a power invocation.

        Only a requested derivative is calculated, and a derivative's domain
        check runs only when the derivative it guards was requested.
        """
        base, exponent = inputs
        need_base, need_exponent = needs_input_grad
        # No operand is inspected here. The exponent-gradient domain checks
        # that used to stand in this place read both tensors to the host,
        # raised for a negative or zero base, and in raising discarded the
        # *base* gradient as well — the case section 12.7.3 works through.
        # Rules G1 to G3 replace all three behaviours: each requested gradient
        # is computed on its own, an absent derivative is NaN rather than an
        # exception, and no host synchronisation detects any of it.
        base_grad = None
        if need_base:
            storage = execute_power_base_gradient(grad, base, exponent)
            # Rule G5: each gradient carries its own operand's declared dtype,
            # not the upstream gradient's.
            base_grad = Tensor._from_owned_storage(
                storage, dtype=base.dtype, shape=grad.shape
            )
        exponent_grad = None
        if need_exponent:
            storage = execute_power_exponent_gradient(grad, base, exponent)
            exponent_grad = Tensor._from_owned_storage(
                storage, dtype=exponent.dtype, shape=grad.shape
            )
        return [
            sum_to_shape(base_grad, base.shape) if base_grad is not None else None,
            (
                sum_to_shape(exponent_grad, exponent.shape)
                if exponent_grad is not None
                else None
            ),
        ]

    def backward_graph(self, grad, *inputs, needs_input_grad: tuple[bool, ...]):
        """Build the requested differentiable VJPs for exponentiation."""
        base, exponent = inputs
        need_base, need_exponent = needs_input_grad
        # The same three corrections as in `backward`; the differentiable path
        # must implement the same D7 semantics as the eager one.
        return [
            (
                sum_to_shape_graph(_power_base_vjp(grad, base, exponent), base.shape)
                if need_base
                else None
            ),
            (
                sum_to_shape_graph(
                    _power_exponent_vjp(grad, base, exponent), exponent.shape
                )
                if need_exponent
                else None
            ),
        ]


def _reduced(
    values: list[float], reference: Tensor, shape: Shape, target: Tensor
) -> Tensor:
    """Reduce accumulated per-element VJP values back to an operand shape."""
    return sum_to_shape(
        Tensor(values, dtype=reference.dtype, shape=shape), target.shape
    )


def _expanded_power_inputs(
    grad: Tensor, base: Tensor, exponent: Tensor
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

    def forward(self, grad: Tensor, base: Tensor, exponent: Tensor) -> Tensor:
        grad, base, exponent = _expanded_power_inputs(grad, base, exponent)
        accelerated = execute_power_base_gradient(grad, base, exponent)
        # Rule G5: the base's declared dtype, not the upstream gradient's.
        return Tensor._from_owned_storage(
            accelerated, dtype=base.dtype, shape=grad.shape
        )

    def backward(
        self, outer_grad: Tensor, *inputs: Tensor, needs_input_grad: tuple[bool, ...]
    ) -> List[Optional[Tensor]]:
        grad, base, exponent = inputs
        need_grad, need_base, need_exponent = needs_input_grad
        expanded_grad, expanded_base, expanded_exponent = _expanded_power_inputs(
            grad, base, exponent
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
                    _base_gradient_value(
                        outer, base_value, power, result, _SMALLEST_NORMAL
                    )
                )
            if need_base:
                base_values.append(
                    _power_product(
                        [outer, upstream, power, power - 1.0], base_value, power - 2.0
                    )
                )
            if not need_exponent:
                exponent_values.append(0.0)
            elif base_value == 0.0:
                if power > 1.0:
                    exponent_values.append(0.0)
                else:
                    raise ValueError(
                        "higher-order power derivatives are undefined at this zero base"
                    )
            else:
                coefficient = math.fsum([1.0, power * math.log(base_value)])
                exponent_values.append(
                    _power_product(
                        [outer, upstream, coefficient], base_value, power - 1.0
                    )
                )
        shape = expanded_grad.shape
        return [
            _reduced(grad_values, outer_grad, shape, grad) if need_grad else None,
            _reduced(base_values, outer_grad, shape, base) if need_base else None,
            (
                _reduced(exponent_values, outer_grad, shape, exponent)
                if need_exponent
                else None
            ),
        ]

    def backward_graph(self, outer_grad, *inputs, needs_input_grad: tuple[bool, ...]):
        from tensors.operations.elementary.log import log

        grad, base, exponent = inputs
        need_grad, need_base, need_exponent = needs_input_grad
        return [
            (
                sum_to_shape_graph(
                    _power_base_vjp(outer_grad, base, exponent), grad.shape
                )
                if need_grad
                else None
            ),
            (
                sum_to_shape_graph(
                    outer_grad
                    * grad
                    * exponent
                    * (exponent - 1.0)
                    * base ** (exponent - 2.0),
                    base.shape,
                )
                if need_base
                else None
            ),
            (
                sum_to_shape_graph(
                    outer_grad
                    * grad
                    * base ** (exponent - 1.0)
                    * (1.0 + exponent * log(base)),
                    exponent.shape,
                )
                if need_exponent
                else None
            ),
        ]


class PowerExponentGradient(Operation):
    """Differentiable range-safe VJP with respect to a power exponent."""

    __slots__ = ()
    name = "power_exponent_gradient"

    def forward(self, grad: Tensor, base: Tensor, exponent: Tensor) -> Tensor:
        grad, base, exponent = _expanded_power_inputs(grad, base, exponent)
        accelerated = execute_power_exponent_gradient(grad, base, exponent)
        # Rule G5: the exponent's declared dtype.
        return Tensor._from_owned_storage(
            accelerated, dtype=exponent.dtype, shape=grad.shape
        )

    def backward(
        self, outer_grad: Tensor, *inputs: Tensor, needs_input_grad: tuple[bool, ...]
    ) -> List[Optional[Tensor]]:
        grad, base, exponent = inputs
        need_grad, need_base, need_exponent = needs_input_grad
        expanded_grad, expanded_base, expanded_exponent = _expanded_power_inputs(
            grad, base, exponent
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
                    _exponent_gradient_value(outer, result, base_value, power)
                )
            if base_value == 0.0:
                if not need_base or power > 1.0:
                    base_values.append(0.0)
                    exponent_values.append(0.0)
                    continue
                raise ValueError(
                    "higher-order power derivatives are undefined at this zero base"
                )
            logarithm = math.log(base_value)
            if need_base:
                coefficient = math.fsum([1.0, power * logarithm])
                base_values.append(
                    _power_product(
                        [outer, upstream, coefficient], base_value, power - 1.0
                    )
                )
            if need_exponent:
                exponent_values.append(
                    _power_product(
                        [outer, upstream, logarithm, logarithm], base_value, power
                    )
                )
        shape = expanded_grad.shape
        return [
            _reduced(grad_values, outer_grad, shape, grad) if need_grad else None,
            _reduced(base_values, outer_grad, shape, base) if need_base else None,
            (
                _reduced(exponent_values, outer_grad, shape, exponent)
                if need_exponent
                else None
            ),
        ]

    def backward_graph(self, outer_grad, *inputs, needs_input_grad: tuple[bool, ...]):
        from tensors.operations.elementary.log import log

        grad, base, exponent = inputs
        need_grad, need_base, need_exponent = needs_input_grad
        logarithm = log(base)
        return [
            (
                sum_to_shape_graph(
                    _power_exponent_vjp(outer_grad, base, exponent), grad.shape
                )
                if need_grad
                else None
            ),
            (
                sum_to_shape_graph(
                    outer_grad
                    * grad
                    * base ** (exponent - 1.0)
                    * (1.0 + exponent * logarithm),
                    base.shape,
                )
                if need_base
                else None
            ),
            (
                sum_to_shape_graph(
                    outer_grad * grad * base**exponent * logarithm**2.0, exponent.shape
                )
                if need_exponent
                else None
            ),
        ]


def _power_base_vjp(grad, base, exponent):
    from tensors.variable import Variable

    operation = PowerBaseGradient()
    return Variable._apply_operation(operation, (grad, base, exponent))


def _power_exponent_vjp(grad, base, exponent):
    from tensors.variable import Variable

    operation = PowerExponentGradient()
    return Variable._apply_operation(operation, (grad, base, exponent))


@overload
def pow(base: Variable, exponent: TensorLike) -> Variable: ...


@overload
def pow(base: TensorLike, exponent: Variable) -> Variable: ...


@overload
def pow(base: TensorData, exponent: TensorData) -> Tensor: ...


def pow(base: TensorLike, exponent: TensorLike) -> TensorResult:
    """Return the element-wise power of two Tensors, Variables, or scalars."""
    return base**exponent


_power_values = Pow().forward
power = _power_values


def power_scalar_base(base: Scalar, exponent: Tensor) -> Tensor:
    """Return ``base`` raised element-wise to ``exponent`` for a scalar base.

    The scalar base converts under rule S-p (section 12.5.3), which reads no
    element of ``exponent``.
    """
    dtype, base = resolve_power_scalar_base(base, exponent.dtype)
    _reject_negative_exponent(dtype, exponent)
    accelerated = execute_power(
        base, exponent, dtype=dtype, output_shape=exponent.shape
    )
    return Tensor._from_owned_storage(accelerated, dtype=dtype, shape=exponent.shape)


__all__ = ["Pow", "pow", "power", "power_scalar_base"]

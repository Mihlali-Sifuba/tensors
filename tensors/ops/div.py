"""Division operation."""

from typing import List, Optional, Union

from ..backend import execute_binary, execute_division_denominator_gradient
from ..dtype import result_dtype
from ..graph.operation import Operation
from ..tensor import Tensor
from ..utils.broadcasting import (
    broadcast_binary_values,
    broadcast_to,
    broadcast_tensors,
)
from ._utils import sum_to_shape


Scalar = Union[int, float]


def _negative_product_over_square(
    left: float,
    right: float,
    denominator: float,
) -> float:
    """Evaluate ``-left * right / denominator**2`` without range loss."""
    return _product_over_denominator_power(
        [-float(left), float(right)],
        float(denominator),
        2,
    )


def _product_over_denominator_power(
    factors: list[float],
    denominator: float,
    power: int,
) -> float:
    """Evaluate a product divided by a denominator power exactly when finite."""
    import math

    denominator = float(denominator)
    if denominator == 0.0:
        raise ZeroDivisionError("Division by zero")
    if any(value == 0.0 for value in factors):
        return 0.0
    if all(math.isfinite(value) for value in factors + [denominator]):
        numerator = 1
        divisor = 1
        for factor in factors:
            factor_numerator, factor_denominator = factor.as_integer_ratio()
            numerator *= factor_numerator
            divisor *= factor_denominator
        denominator_numerator, denominator_denominator = (
            denominator.as_integer_ratio()
        )
        numerator *= denominator_denominator ** power
        divisor *= denominator_numerator ** power
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


class Div(Operation):
    """Element-wise division — forward and backward."""

    __slots__ = ()
    name = "div"

    def forward(self, a: Tensor, b: Union[Tensor, Scalar]) -> Tensor:
        """Element-wise division."""
        if not isinstance(b, (int, float, Tensor)):
            raise TypeError(f"Unsupported: {type(b)}")
        dtype = result_dtype(a.dtype, b, division=True)
        if isinstance(b, (int, float)):
            if b == 0:
                raise ZeroDivisionError("Division by zero")
            accelerated = execute_binary(
                "divide",
                a,
                b,
                dtype=dtype,
                output_shape=a.shape,
            )
            if accelerated is not None:
                return Tensor._from_owned_storage(accelerated, dtype=dtype, shape=a.shape)
            data = [x / b for x in a._data]
            return Tensor._from_values(data, dtype, a.shape)
        if isinstance(b, Tensor):
            shape = a.shape.broadcast_with(b.shape)
            accelerated = execute_binary(
                "divide",
                a,
                b,
                dtype=dtype,
                output_shape=shape,
            )
            if accelerated is not None:
                return Tensor._from_owned_storage(accelerated, dtype=dtype, shape=shape)
            def divide(x, y):
                if y == 0:
                    raise ZeroDivisionError("Division by zero")
                return x / y

            data = broadcast_binary_values(a, b, shape, divide)
            return Tensor._from_values(data, dtype, shape)
        raise TypeError(f"Unsupported: {type(b)}")

    def backward(
        self,
        grad: Tensor,
        *inputs: Tensor,
        needs_input_grad: tuple[bool, ...],
    ) -> List[Optional[Tensor]]:
        a, b = inputs
        need_numerator, need_denominator = needs_input_grad
        # The denominator VJP needs both operands broadcast together; the
        # numerator VJP does not, so only pay for it when it is requested.
        numerator_gradient = (
            sum_to_shape(self.forward(grad, b), a.shape)
            if need_numerator
            else None
        )
        if not need_denominator:
            return [numerator_gradient, None]

        expanded_a, expanded_b = broadcast_tensors(a, b)
        accelerated = execute_division_denominator_gradient(
            grad,
            expanded_a,
            expanded_b,
        )
        if accelerated is not None:
            db = Tensor._from_owned_storage(accelerated, dtype=grad.dtype, shape=grad.shape)
        else:
            db = Tensor(
                [
                    _negative_product_over_square(g, x, y)
                    for g, x, y in zip(
                        grad._data,
                        expanded_a._data,
                        expanded_b._data,
                    )
                ],
                dtype=grad.dtype,
                shape=grad.shape,
            )
        return [numerator_gradient, sum_to_shape(db, b.shape)]

    def backward_graph(
        self,
        grad,
        *inputs,
        needs_input_grad: tuple[bool, ...],
    ):
        """Build a differentiable VJP for division."""
        left, right = inputs
        need_numerator, need_denominator = needs_input_grad
        from ._utils import sum_to_shape_graph
        return [
            sum_to_shape_graph(grad / right, left.shape)
            if need_numerator
            else None,
            sum_to_shape_graph(
                _division_denominator_vjp(grad, left, right),
                right.shape,
            )
            if need_denominator
            else None,
        ]


def _expanded_division_inputs(
    grad: Tensor,
    numerator: Tensor,
    denominator: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    shape = grad.shape.broadcast_with(numerator.shape).broadcast_with(
        denominator.shape
    )
    return (
        broadcast_to(grad, shape),
        broadcast_to(numerator, shape),
        broadcast_to(denominator, shape),
    )


class DivisionDenominatorGradient(Operation):
    """Differentiable range-safe VJP for a division denominator."""

    __slots__ = ()
    name = "division_denominator_gradient"

    def forward(
        self,
        grad: Tensor,
        numerator: Tensor,
        denominator: Tensor,
    ) -> Tensor:
        grad, numerator, denominator = _expanded_division_inputs(
            grad,
            numerator,
            denominator,
        )
        if any(value == 0 for value in denominator._data):
            raise ZeroDivisionError("Division by zero")
        accelerated = execute_division_denominator_gradient(
            grad,
            numerator,
            denominator,
        )
        if accelerated is not None:
            return Tensor._from_owned_storage(
                accelerated,
                dtype=grad.dtype,
                shape=grad.shape,
            )
        values = [
            _negative_product_over_square(upstream, value, divisor)
            for upstream, value, divisor in zip(
                grad._data,
                numerator._data,
                denominator._data,
            )
        ]
        return Tensor(values, dtype=grad.dtype, shape=grad.shape)

    def backward(
        self,
        outer_grad: Tensor,
        *inputs: Tensor,
        needs_input_grad: tuple[bool, ...],
    ) -> List[Optional[Tensor]]:
        grad, numerator, denominator = inputs
        need_grad, need_numerator, need_denominator = needs_input_grad
        expanded_grad, expanded_numerator, expanded_denominator = (
            _expanded_division_inputs(grad, numerator, denominator)
        )
        expanded_outer = broadcast_to(outer_grad, expanded_grad.shape)
        grad_values = []
        numerator_values = []
        denominator_values = []
        for outer, upstream, value, divisor in zip(
            expanded_outer._data,
            expanded_grad._data,
            expanded_numerator._data,
            expanded_denominator._data,
        ):
            if need_grad:
                grad_values.append(
                    _negative_product_over_square(outer, value, divisor)
                )
            if need_numerator:
                numerator_values.append(
                    _negative_product_over_square(outer, upstream, divisor)
                )
            if not need_denominator:
                continue
            denominator_values.append(
                _product_over_denominator_power(
                    [2.0, float(outer), float(upstream), float(value)],
                    float(divisor),
                    3,
                )
            )
        shape = expanded_grad.shape

        def reduced(values: list[float], target: Tensor) -> Tensor:
            return sum_to_shape(
                Tensor(values, dtype=outer_grad.dtype, shape=shape),
                target.shape,
            )

        return [
            reduced(grad_values, grad) if need_grad else None,
            reduced(numerator_values, numerator) if need_numerator else None,
            reduced(denominator_values, denominator)
            if need_denominator
            else None,
        ]

    def backward_graph(
        self,
        outer_grad,
        *inputs,
        needs_input_grad: tuple[bool, ...],
    ):
        from ._utils import sum_to_shape_graph

        grad, numerator, denominator = inputs
        need_grad, need_numerator, need_denominator = needs_input_grad
        return [
            sum_to_shape_graph(
                _division_denominator_vjp(
                    outer_grad,
                    numerator,
                    denominator,
                ),
                grad.shape,
            )
            if need_grad
            else None,
            sum_to_shape_graph(
                _division_denominator_vjp(
                    outer_grad,
                    grad,
                    denominator,
                ),
                numerator.shape,
            )
            if need_numerator
            else None,
            sum_to_shape_graph(
                2.0
                * outer_grad
                * grad
                * numerator
                / (denominator ** 3.0),
                denominator.shape,
            )
            if need_denominator
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


def _division_denominator_vjp(grad, numerator, denominator):
    from ..variable import Variable

    operation = DivisionDenominatorGradient()
    return Variable._from_operation(
        operation.forward(grad.data, numerator.data, denominator.data),
        operation,
        (grad, numerator, denominator),
    )


divide = Div().forward


def divide_scalar(numerator: Scalar, denominator: Tensor) -> Tensor:
    """Return ``numerator / denominator`` for a scalar left operand."""
    dtype = result_dtype(denominator.dtype, numerator, division=True)
    accelerated = execute_binary(
        "divide",
        numerator,
        denominator,
        dtype=dtype,
        output_shape=denominator.shape,
    )
    if accelerated is not None:
        return Tensor._from_owned_storage(
            accelerated,
            dtype=dtype,
            shape=denominator.shape,
        )
    values = []
    for value in denominator._data:
        if value == 0:
            raise ZeroDivisionError("Division by zero")
        values.append(numerator / value)
    return Tensor(values, dtype=dtype, shape=denominator.shape)

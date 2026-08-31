"""Division operation."""

from typing import List, Union

from ..backend import execute_binary, execute_division_denominator_gradient
from ..dtype import result_dtype
from ..tensor import Tensor
from ..utils.broadcasting import broadcast_to, broadcast_tensors
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


class Div:
    """Element-wise division — forward and backward."""

    @staticmethod
    def forward(a: Tensor, b: Union[Tensor, Scalar]) -> Tensor:
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
                return Tensor(accelerated, dtype=dtype, shape=a.shape)
            data = [x / b for x in a._data]
            return Tensor(data, dtype=dtype, shape=a.shape)
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
                return Tensor(accelerated, dtype=dtype, shape=shape)
            a, b = broadcast_tensors(a, b)
            data = []
            for x, y in zip(a._data, b._data):
                if y == 0:
                    raise ZeroDivisionError("Division by zero")
                data.append(x / y)
            return Tensor(data, dtype=dtype, shape=a.shape)
        raise TypeError(f"Unsupported: {type(b)}")

    @staticmethod
    def _mul_tensors(a: Tensor, b: Tensor) -> Tensor:
        """Element-wise multiply two Tensors (helper for backward)."""
        data = [x * y for x, y in zip(a._data, b._data)]
        return Tensor(data, dtype=a.dtype, shape=a.shape)

    @staticmethod
    def _neg_tensor(t: Tensor) -> Tensor:
        """Negate a Tensor (helper for backward)."""
        return Tensor([-x for x in t._data], dtype=t.dtype.typecode, shape=t.shape)

    @staticmethod
    def forward_reverse(a: Tensor, scalar: Scalar) -> Tensor:
        """Forward for scalar / tensor (reverse division)."""
        dtype = result_dtype(a.dtype, scalar, division=True)
        accelerated = execute_binary(
            "divide",
            scalar,
            a,
            dtype=dtype,
            output_shape=a.shape,
        )
        if accelerated is not None:
            return Tensor(accelerated, dtype=dtype, shape=a.shape)
        values = []
        for denominator in a._data:
            if denominator == 0:
                raise ZeroDivisionError("Division by zero")
            values.append(scalar / denominator)
        return Tensor(values, dtype=dtype, shape=a.shape)

    @staticmethod
    def backward(grad: Tensor, *inputs: Tensor, **kwargs: object) -> List[Tensor]:
        if len(inputs) == 2:
            a, b = inputs
            expanded_a, expanded_b = broadcast_tensors(a, b)
            da = Div.forward(grad, expanded_b)
            accelerated = execute_division_denominator_gradient(
                grad,
                expanded_a,
                expanded_b,
            )
            if accelerated is not None:
                db = Tensor(accelerated, dtype=grad.dtype, shape=grad.shape)
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
            return [sum_to_shape(da, a.shape), sum_to_shape(db, b.shape)]
        scalar = kwargs.get("scalar", 1.0)
        assert isinstance(scalar, (int, float))
        if kwargs.get("reverse", False):
            a = inputs[0]
            dtype = result_dtype(grad.dtype, a, division=True)
            values = (
                _negative_product_over_square(g, scalar, x)
                for g, x in zip(grad._data, a._data)
            )
            return [Tensor(list(values), dtype=dtype, shape=a.shape)]
        return [Div.forward(grad, scalar)]

    @staticmethod
    def backward_graph(grad, *inputs, **kwargs: object):
        """Build a differentiable VJP for division."""
        if len(inputs) == 1:
            scalar = kwargs.get("scalar", 1.0)
            if kwargs.get("reverse", False):
                value = inputs[0]
                numerator = _constant_like(value, float(scalar))
                return [_division_denominator_vjp(grad, numerator, value)]
            return [grad / scalar]
        left, right = inputs
        from ._utils import sum_to_shape_graph
        return [
            sum_to_shape_graph(grad / right, left.shape),
            sum_to_shape_graph(
                _division_denominator_vjp(grad, left, right),
                right.shape,
            ),
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


class DivisionDenominatorGradient:
    """Differentiable range-safe VJP for a division denominator."""

    @staticmethod
    def forward(
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
            return Tensor(
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

    @staticmethod
    def backward(
        outer_grad: Tensor,
        *inputs: Tensor,
        **kwargs: object,
    ) -> List[Tensor]:
        grad, numerator, denominator = inputs
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
            grad_values.append(
                _negative_product_over_square(outer, value, divisor)
            )
            numerator_values.append(
                _negative_product_over_square(outer, upstream, divisor)
            )
            denominator_values.append(
                _product_over_denominator_power(
                    [2.0, float(outer), float(upstream), float(value)],
                    float(divisor),
                    3,
                )
            )
        shape = expanded_grad.shape
        return [
            sum_to_shape(
                Tensor(grad_values, dtype=outer_grad.dtype, shape=shape),
                grad.shape,
            ),
            sum_to_shape(
                Tensor(numerator_values, dtype=outer_grad.dtype, shape=shape),
                numerator.shape,
            ),
            sum_to_shape(
                Tensor(denominator_values, dtype=outer_grad.dtype, shape=shape),
                denominator.shape,
            ),
        ]

    @staticmethod
    def backward_graph(outer_grad, *inputs, **kwargs: object):
        from ._utils import sum_to_shape_graph

        grad, numerator, denominator = inputs
        return [
            sum_to_shape_graph(
                _division_denominator_vjp(
                    outer_grad,
                    numerator,
                    denominator,
                ),
                grad.shape,
            ),
            sum_to_shape_graph(
                _division_denominator_vjp(
                    outer_grad,
                    grad,
                    denominator,
                ),
                numerator.shape,
            ),
            sum_to_shape_graph(
                2.0
                * outer_grad
                * grad
                * numerator
                / (denominator ** 3.0),
                denominator.shape,
            ),
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

    return Variable._from_operation(
        DivisionDenominatorGradient.forward(
            grad.data,
            numerator.data,
            denominator.data,
        ),
        "division_denominator_gradient",
        DivisionDenominatorGradient,
        [grad, numerator, denominator],
    )

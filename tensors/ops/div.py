"""Division operation."""

from typing import List, Union

from ..dtype import result_dtype
from ..tensor import Tensor
from ..utils.broadcasting import broadcast_tensors
from ._utils import sum_to_shape


Scalar = Union[int, float]


def _negative_product_over_square(
    left: float,
    right: float,
    denominator: float,
) -> float:
    """Evaluate ``-left * right / denominator**2`` without range loss."""
    import math

    left = float(left)
    right = float(right)
    denominator = float(denominator)
    if all(math.isfinite(value) for value in (left, right, denominator)):
        if denominator == 0.0:
            raise ZeroDivisionError("Division by zero")
        left_numerator, left_denominator = left.as_integer_ratio()
        right_numerator, right_denominator = right.as_integer_ratio()
        denominator_numerator, denominator_denominator = (
            denominator.as_integer_ratio()
        )
        numerator = (
            -left_numerator
            * right_numerator
            * denominator_denominator
            * denominator_denominator
        )
        divisor = (
            left_denominator
            * right_denominator
            * denominator_numerator
            * denominator_numerator
        )
        try:
            return numerator / divisor
        except OverflowError:
            return math.inf if numerator > 0 else -math.inf
    return -(left / denominator) * (right / denominator)


class Div:
    """Element-wise division — forward and backward."""

    @staticmethod
    def forward(a: Tensor, b: Union[Tensor, Scalar]) -> Tensor:
        """Element-wise division."""
        dtype = result_dtype(a.dtype, b, division=True)
        if isinstance(b, (int, float)):
            if b == 0:
                raise ZeroDivisionError("Division by zero")
            data = [x / b for x in a._data]
            return Tensor(data, dtype=dtype, shape=a.shape)
        if isinstance(b, Tensor):
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
            da = Tensor(
                [g / y for g, y in zip(grad._data, expanded_b._data)],
                dtype=grad.dtype,
                shape=grad.shape,
            )
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
        return [Tensor([g / scalar for g in grad._data], dtype=grad.dtype.typecode, shape=grad.shape)]

    @staticmethod
    def backward_graph(grad, *inputs, **kwargs: object):
        """Build a differentiable VJP for division."""
        if len(inputs) == 1:
            scalar = kwargs.get("scalar", 1.0)
            if kwargs.get("reverse", False):
                value = inputs[0]
                return [-(grad / value) * (scalar / value)]
            return [grad / scalar]
        left, right = inputs
        from ._utils import sum_to_shape_graph
        return [
            sum_to_shape_graph(grad / right, left.shape),
            sum_to_shape_graph(-(grad / right) * (left / right), right.shape),
        ]

"""Sum and its differentiation rule."""

import builtins
import math
from typing import Any, List

from ..tensor import Tensor
from ._reduction import Axis, immutable_axis, keepdims_shape, reduction_groups


def _sum_exact_ratios(
    ratios: list[tuple[int, int]],
    *,
    divisor: int = 1,
) -> float:
    """Convert an exact sum of binary ratios, optionally divided, to a float."""
    denominator = max((item[1] for item in ratios), default=1)
    numerator = builtins.sum(
        item_numerator * (denominator // item_denominator)
        for item_numerator, item_denominator in ratios
    )
    try:
        return numerator / (denominator * divisor)
    except OverflowError:
        return math.inf if numerator > 0 else -math.inf


def _stable_float_sum(values: list[float]) -> float:
    """Sum floats accurately even when a temporary partial sum overflows."""
    if any(math.isnan(value) for value in values):
        return math.nan
    has_positive_infinity = math.inf in values
    has_negative_infinity = -math.inf in values
    if has_positive_infinity and has_negative_infinity:
        return math.nan
    if has_positive_infinity:
        return math.inf
    if has_negative_infinity:
        return -math.inf

    try:
        return math.fsum(values)
    except OverflowError:
        return _sum_exact_ratios(
            [value.as_integer_ratio() for value in values]
        )


def _stable_product_sum(factors: list[tuple[float, float]]) -> float:
    """Accurately sum products, including products outside float range."""
    products = [left * right for left, right in factors]
    finite_factors = all(
        math.isfinite(left) and math.isfinite(right)
        for left, right in factors
    )
    product_lost_range = any(
        (math.isinf(product) or (
            product == 0.0 and left != 0.0 and right != 0.0
        ))
        for (left, right), product in zip(factors, products)
    )
    if not finite_factors or not product_lost_range:
        return _stable_float_sum(products)

    ratios = []
    for left, right in factors:
        left_numerator, left_denominator = left.as_integer_ratio()
        right_numerator, right_denominator = right.as_integer_ratio()
        ratios.append((
            left_numerator * right_numerator,
            left_denominator * right_denominator,
        ))
    return _sum_exact_ratios(ratios)


def _sum_impl(a: Tensor, axis: Axis = None,
              keepdims: bool = False) -> Tensor:
    """Sum over one, several, or all axes."""
    _, output_shape, groups = reduction_groups(
        a, axis, keepdims, scalar_as_vector=True
    )
    if a.dtype.kind == "floating":
        values = [
            _stable_float_sum([float(a._data[index]) for index in group])
            for group in groups
        ]
    else:
        values = [
            builtins.sum(a._data[index] for index in group)
            for group in groups
        ]
    return Tensor(values, dtype=a.dtype, shape=output_shape)


class Sum:
    """Sum with a reverse-mode gradient rule."""

    @staticmethod
    def forward(a: Tensor, axis: Axis = None,
                keepdims: bool = False) -> Tensor:
        return _sum_impl(a, axis=axis, keepdims=keepdims)

    @staticmethod
    def backward(grad: Tensor, *inputs: Tensor, **kwargs: object) -> List[Tensor]:
        a = inputs[0]
        axis = kwargs.get("axis")
        keepdims = kwargs.get("keepdims", False)
        _, output_shape, groups = reduction_groups(
            a, axis, keepdims, scalar_as_vector=True
        )
        if grad.shape != output_shape:
            raise ValueError(
                f"Gradient shape {grad.shape} does not match output shape {output_shape}"
            )
        result = [0.0] * a.size
        for output_index, group in enumerate(groups):
            for input_index in group:
                result[input_index] = grad._data[output_index]

        return [Tensor(result, dtype=grad.dtype, shape=a.shape)]

    @staticmethod
    def backward_graph(grad, *inputs, **kwargs: object):
        """Build a differentiable VJP for an axis-aware sum."""
        from ..variable import Variable
        from .reshape import reshape

        axis = kwargs.get("axis")
        keepdims = kwargs.get("keepdims", False)
        value = inputs[0]
        expanded = grad if keepdims else reshape(grad, keepdims_shape(value.shape, axis))
        ones = Variable(
            Tensor([1.0] * value.size, dtype=grad.dtype, shape=value.shape),
            requires_grad=False,
        )
        return [expanded * ones]


def sum(value: Any, axis: Axis = None,
        keepdims: bool = False) -> Any:
    """Sum over one, several, or all axes."""
    from ..variable import Variable

    axis = immutable_axis(axis)

    if isinstance(value, Variable):
        return Variable._from_operation(
            Sum.forward(value.data, axis=axis, keepdims=keepdims),
            "sum",
            Sum,
            [value],
            axis=axis,
            keepdims=keepdims,
        )
    if not isinstance(value, Tensor):
        value = Tensor(value)
    return Sum.forward(value, axis=axis, keepdims=keepdims)


__all__ = ["Sum", "sum"]

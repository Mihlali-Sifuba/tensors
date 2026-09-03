"""Sum and its differentiation rule."""

import builtins
import math
from typing import Any, List, overload

from .._typing import TensorData, TensorLike, TensorResult, TensorValue
from ..backend import execute_reduction, execute_reduction_gradient
from ..graph.operation import Operation
from ..tensor import Tensor
from ._reduction import (
    Axis,
    immutable_axis,
    keepdims_shape,
    normalize_axes,
    reduction_groups,
    reduction_shape,
)


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
    if len(factors) == 1:
        # A single finite, non-zero product neither cancels nor loses range,
        # so it is already the exact result the general path would return.
        left, right = factors[0]
        product = left * right
        if product and math.isfinite(product):
            return product

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
    axes = normalize_axes(a.ndim, axis)
    output_shape = reduction_shape(a.shape, axes, keepdims)
    if axis is None and not keepdims:
        output_shape = (1,)
    accelerated = execute_reduction(
        "sum",
        a,
        axes,
        keepdims=keepdims,
        dtype=a.dtype,
        output_shape=output_shape,
    )
    if accelerated is not None:
        return Tensor._from_owned_storage(accelerated, dtype=a.dtype, shape=output_shape)
    data = a._data
    if axes == tuple(range(a.ndim)):
        if a.dtype.kind == "floating":
            total = _stable_float_sum([float(value) for value in data])
        else:
            total = builtins.sum(data)
        return Tensor([total], dtype=a.dtype, shape=output_shape)
    _, output_shape, groups = reduction_groups(
        a, axis, keepdims, scalar_as_vector=True
    )
    if a.dtype.kind == "floating":
        values = [
            _stable_float_sum([float(data[index]) for index in group])
            for group in groups
        ]
    else:
        values = [
            builtins.sum(data[index] for index in group)
            for group in groups
        ]
    return Tensor(values, dtype=a.dtype, shape=output_shape)


class Sum(Operation):
    """Sum with a reverse-mode gradient rule."""

    __slots__ = ("axis", "keepdims")
    name = "sum"

    def __init__(
        self,
        *,
        axis: Axis = None,
        keepdims: bool = False,
    ) -> None:
        object.__setattr__(self, "axis", axis)
        object.__setattr__(self, "keepdims", keepdims)

    def forward(self, a: Tensor) -> Tensor:
        axis = self.axis
        keepdims = self.keepdims
        return _sum_impl(a, axis=axis, keepdims=keepdims)

    def backward(self, grad: Tensor, *inputs: Tensor) -> List[Tensor]:
        a = inputs[0]
        axis = self.axis
        keepdims = self.keepdims
        axes = normalize_axes(a.ndim, axis)
        output_shape = reduction_shape(a.shape, axes, keepdims)
        if axis is None and not keepdims:
            output_shape = (1,)
        if grad.shape != output_shape:
            raise ValueError(
                f"Gradient shape {grad.shape} does not match output shape {output_shape}"
            )
        accelerated = execute_reduction_gradient(
            "sum",
            grad,
            a,
            axes,
            keepdims=keepdims,
        )
        if accelerated is not None:
            return [Tensor._from_owned_storage(accelerated, dtype=grad.dtype, shape=a.shape)]
        _, _, groups = reduction_groups(
            a, axis, keepdims, scalar_as_vector=True
        )
        result = [0.0] * a.size
        for output_index, group in enumerate(groups):
            for input_index in group:
                result[input_index] = grad._data[output_index]

        return [Tensor(result, dtype=grad.dtype, shape=a.shape)]

    def backward_graph(self, grad, *inputs):
        """Build a differentiable VJP for an axis-aware sum."""
        from ..creation import ones
        from ..variable import Variable
        from .reshape import reshape

        axis = self.axis
        keepdims = self.keepdims
        value = inputs[0]
        expanded = grad if keepdims else reshape(grad, keepdims_shape(value.shape, axis))
        unit = Variable(
            ones(value.shape, dtype=grad.dtype),
            requires_grad=False,
        )
        return [expanded * unit]


@overload
def sum(
    value: TensorValue,
    axis: Axis = None,
    keepdims: bool = False,
) -> TensorValue: ...


@overload
def sum(
    value: TensorData,
    axis: Axis = None,
    keepdims: bool = False,
) -> Tensor: ...


def sum(
    value: TensorLike,
    axis: Axis = None,
    keepdims: bool = False,
) -> TensorResult:
    """Sum over one, several, or all axes."""
    from ..variable import Variable

    axis = immutable_axis(axis)

    if isinstance(value, Variable):
        operation = Sum(axis=axis, keepdims=keepdims)
        return Variable._from_operation(
            operation.forward(value.data),
            operation,
            (value,),
        )
    if not isinstance(value, Tensor):
        value = Tensor(value)
    return Sum(axis=axis, keepdims=keepdims).forward(value)


__all__ = ["Sum", "sum"]

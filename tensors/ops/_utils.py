"""Shared gradient-shape helpers for differentiable operations."""

from __future__ import annotations

from ..shape import Shape
from .operation import Operation
from ..tensor import Tensor
from ..utils.coordinates import (
    coordinates_to_linear_index,
    linear_index_to_coordinates,
)


def sum_to_shape(gradient: Tensor, shape: tuple[int, ...]) -> Tensor:
    """Reduce a broadcasted ``gradient`` back to an input ``shape``.

    During a forward broadcast, a value can participate in several output
    positions. Reverse-mode differentiation must sum those contributions into
    the corresponding original position.
    """
    if gradient.shape == shape:
        return gradient
    if len(shape) > gradient.ndim:
        raise ValueError(f"Cannot reduce gradient shape {gradient.shape} to {shape}")

    from ..backend import execute_sum_to_shape

    accelerated = execute_sum_to_shape(gradient, shape)
    if accelerated is not None:
        return Tensor._from_owned_storage(accelerated, dtype=gradient.dtype, shape=shape)

    target = Shape.from_iterable(shape)
    if target.size == 1:
        if gradient.dtype.kind == "floating":
            from ..math.sum import _stable_float_sum

            total = _stable_float_sum([float(value) for value in gradient._data])
        else:
            total = sum(gradient._data)
        return Tensor._from_values([total], gradient.dtype, target)

    padded_shape = (1,) * (gradient.ndim - len(shape)) + shape
    groups: list[list[int | float]] = [
        [] for _ in range(target.size)
    ]
    padding = gradient.ndim - len(shape)
    for index, value in enumerate(gradient._data):
        gradient_coordinates = linear_index_to_coordinates(
            index,
            gradient.shape,
        )
        source_coordinates = tuple(
            0 if source_dimension == 1 else coordinate
            for source_dimension, coordinate in zip(padded_shape, gradient_coordinates)
        )[padding:]
        groups[
            coordinates_to_linear_index(source_coordinates, shape)
        ].append(value)

    if gradient.dtype.kind == "floating":
        from ..math.sum import _stable_float_sum

        values = [
            _stable_float_sum([float(value) for value in group])
            for group in groups
        ]
    else:
        values = [sum(group) for group in groups]

    return Tensor(values, dtype=gradient.dtype, shape=shape)


def sum_products_to_shape(
    gradient: Tensor,
    factor: Tensor,
    shape: tuple[int, ...],
) -> Tensor:
    """Fused multiply-and-reduce for a broadcast VJP.

    Multiplying first can create opposite infinities even when the exact
    reduced result is finite. Grouping the factors before rounding preserves
    that cancellation.
    """
    from ..backend import execute_sum_products_to_shape
    from ..math.sum import _stable_product_sum
    from ..utils.broadcasting import broadcast_tensors

    accelerated = execute_sum_products_to_shape(gradient, factor, shape)
    if accelerated is not None:
        return Tensor._from_owned_storage(accelerated, dtype=gradient.dtype, shape=shape)

    expanded_gradient, expanded_factor = broadcast_tensors(gradient, factor)
    if len(shape) > expanded_gradient.ndim:
        raise ValueError(
            f"Cannot reduce gradient shape {expanded_gradient.shape} to {shape}"
        )

    # Both grouping shortcuts still round through _stable_product_sum, so the
    # cancellation guarantees are unchanged; only the coordinate mapping is
    # skipped.
    target = Shape.from_iterable(shape)
    if expanded_gradient.shape == target:
        values = [
            _stable_product_sum([(float(left), float(right))])
            for left, right in zip(
                expanded_gradient._data,
                expanded_factor._data,
            )
        ]
        return Tensor._from_values(values, gradient.dtype, target)
    if target.size == 1:
        values = [
            _stable_product_sum([
                (float(left), float(right))
                for left, right in zip(
                    expanded_gradient._data,
                    expanded_factor._data,
                )
            ])
        ]
        return Tensor._from_values(values, gradient.dtype, target)

    padded_shape = (1,) * (expanded_gradient.ndim - len(shape)) + shape
    padding = expanded_gradient.ndim - len(shape)
    groups: list[list[tuple[float, float]]] = [
        [] for _ in range(Shape.from_iterable(shape).size)
    ]
    for index, (left, right) in enumerate(
        zip(expanded_gradient._data, expanded_factor._data)
    ):
        coordinates = linear_index_to_coordinates(
            index,
            expanded_gradient.shape,
        )
        source_coordinates = tuple(
            0 if source_dimension == 1 else coordinate
            for source_dimension, coordinate in zip(padded_shape, coordinates)
        )[padding:]
        source_index = coordinates_to_linear_index(source_coordinates, shape)
        groups[source_index].append((float(left), float(right)))

    values = [_stable_product_sum(group) for group in groups]
    return Tensor(values, dtype=gradient.dtype, shape=shape)


def sum_to_shape_graph(gradient, shape: tuple[int, ...]):
    """Differentiably reduce a broadcasted Variable back to ``shape``."""
    from ..math import reshape, sum

    if gradient.shape == shape:
        return gradient
    if len(shape) > gradient.ndim:
        raise ValueError(f"Cannot reduce gradient shape {gradient.shape} to {shape}")

    padding = gradient.ndim - len(shape)
    padded_shape = (1,) * padding + shape
    axes = tuple(
        axis
        for axis, (source, target) in enumerate(zip(gradient.shape, padded_shape))
        if target == 1 and source != 1
    )
    reduced = sum(gradient, axis=axes, keepdims=True) if axes else gradient
    return reshape(reduced, shape) if reduced.shape != shape else reduced


class ProductSumToShape(Operation):
    """Fused differentiable product reduction used by broadcast VJPs."""

    __slots__ = ("target_shape",)
    name = "product_sum_to_shape"

    def __init__(
        self,
        *,
        target_shape: tuple[int, ...],
    ) -> None:
        object.__setattr__(self, "target_shape", target_shape)

    def forward(self, left: Tensor, right: Tensor) -> Tensor:
        target_shape = self.target_shape
        return sum_products_to_shape(left, right, target_shape)

    def backward(
        self,
        grad: Tensor,
        *inputs: Tensor,
        needs_input_grad: tuple[bool, ...],
    ) -> list[Tensor]:
        from ..utils.broadcasting import broadcast_to

        left, right = inputs
        need_left, need_right = needs_input_grad
        common_shape = left.shape.broadcast_with(right.shape)
        expanded_grad = broadcast_to(grad, common_shape)
        return [
            sum_products_to_shape(expanded_grad, right, left.shape)
            if need_left
            else None,
            sum_products_to_shape(expanded_grad, left, right.shape)
            if need_right
            else None,
        ]

    def backward_graph(
        self,
        grad,
        *inputs,
        needs_input_grad: tuple[bool, ...],
    ):
        left, right = inputs
        common_shape = left.shape.broadcast_with(right.shape)
        ones = Tensor(
            [1.0] * common_shape.size,
            dtype=grad.dtype,
            shape=common_shape,
        )
        expanded_grad = grad * ones
        need_left, need_right = needs_input_grad
        return [
            sum_products_to_shape_graph(expanded_grad, right, left.shape)
            if need_left
            else None,
            sum_products_to_shape_graph(expanded_grad, left, right.shape)
            if need_right
            else None,
        ]


def sum_products_to_shape_graph(left, right, shape: tuple[int, ...]):
    """Record a fused multiply-and-reduce without premature overflow."""
    from ..variable import Variable

    operation = ProductSumToShape(target_shape=shape)
    return Variable._from_operation(
        operation.forward(left.data, right.data),
        operation,
        (left, right),
    )


class ZeroLike(Operation):
    """A graph-connected zero that is safe for infinite input values."""

    __slots__ = ()
    name = "zero_like"

    def forward(self, value: Tensor) -> Tensor:
        return Tensor(
            [0.0] * value.size,
            dtype=value.dtype,
            shape=value.shape,
        )

    def backward(
        self,
        grad: Tensor,
        *inputs: Tensor,
        needs_input_grad: tuple[bool, ...],
    ) -> list[Tensor]:
        return [
            Tensor(
                [0.0] * inputs[0].size,
                dtype=grad.dtype,
                shape=inputs[0].shape,
            )
        ]

    def backward_graph(
        self,
        grad,
        *inputs,
        needs_input_grad: tuple[bool, ...],
    ):
        return [zero_like_graph(inputs[0])]


def zero_like_graph(value):
    """Return graph-connected zeros without evaluating ``value * 0``."""
    from ..variable import Variable

    operation = ZeroLike()
    return Variable._from_operation(
        operation.forward(value.data),
        operation,
        (value,),
    )


class MaskedValue(Operation):
    """Select values with a constant mask without evaluating ``infinity * 0``."""

    __slots__ = ("mask",)
    name = "masked_value"

    def __init__(
        self,
        *,
        mask: Tensor,
    ) -> None:
        object.__setattr__(self, "mask", mask)

    def forward(self, value: Tensor) -> Tensor:
        mask = self.mask
        from ..utils.broadcasting import broadcast_to

        expanded = broadcast_to(value, mask.shape)
        return Tensor(
            [
                item if selected != 0.0 else 0.0
                for item, selected in zip(expanded._data, mask._data)
            ],
            dtype=value.dtype,
            shape=mask.shape,
        )

    def backward(
        self,
        grad: Tensor,
        *inputs: Tensor,
        needs_input_grad: tuple[bool, ...],
    ) -> list[Tensor]:
        mask = self.mask
        if not isinstance(mask, Tensor):
            raise TypeError("masked-value mask must be a Tensor")
        selected = Tensor(
            [
                gradient if selected != 0.0 else 0.0
                for gradient, selected in zip(grad._data, mask._data)
            ],
            dtype=grad.dtype,
            shape=mask.shape,
        )
        return [sum_to_shape(selected, inputs[0].shape)]

    def backward_graph(
        self,
        grad,
        *inputs,
        needs_input_grad: tuple[bool, ...],
    ):
        mask = self.mask
        if not isinstance(mask, Tensor):
            raise TypeError("masked-value mask must be a Tensor")
        return [
            sum_to_shape_graph(
                masked_value_graph(grad, mask),
                inputs[0].shape,
            )
        ]


def masked_value_graph(value, mask: Tensor):
    """Select graph values using a constant zero-one mask."""
    from ..variable import Variable

    if value.shape.broadcast_with(mask.shape) != mask.shape:
        raise ValueError(
            f"Value shape {value.shape} cannot broadcast to mask shape "
            f"{mask.shape}"
        )
    operation = MaskedValue(mask=mask)
    return Variable._from_operation(
        operation.forward(value.data),
        operation,
        (value,),
    )


__all__ = [
    "MaskedValue",
    "ProductSumToShape",
    "masked_value_graph",
    "sum_products_to_shape",
    "sum_products_to_shape_graph",
    "sum_to_shape",
    "sum_to_shape_graph",
    "ZeroLike",
    "zero_like_graph",
]

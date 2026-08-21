"""Shared gradient-shape helpers for differentiable operations."""

from __future__ import annotations

from ..tensor import Tensor
from ..utils.shape import coordinates_to_index, index_to_coordinates, shape_size


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

    padded_shape = (1,) * (gradient.ndim - len(shape)) + shape
    groups: list[list[int | float]] = [
        [] for _ in range(shape_size(shape))
    ]
    padding = gradient.ndim - len(shape)
    for index, value in enumerate(gradient._data):
        gradient_coordinates = index_to_coordinates(index, gradient.shape)
        source_coordinates = tuple(
            0 if source_dimension == 1 else coordinate
            for source_dimension, coordinate in zip(padded_shape, gradient_coordinates)
        )[padding:]
        groups[coordinates_to_index(source_coordinates, shape)].append(value)

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
    from ..math.sum import _stable_product_sum
    from ..utils.broadcasting import broadcast_tensors

    expanded_gradient, expanded_factor = broadcast_tensors(gradient, factor)
    if len(shape) > expanded_gradient.ndim:
        raise ValueError(
            f"Cannot reduce gradient shape {expanded_gradient.shape} to {shape}"
        )

    padded_shape = (1,) * (expanded_gradient.ndim - len(shape)) + shape
    padding = expanded_gradient.ndim - len(shape)
    groups: list[list[tuple[float, float]]] = [
        [] for _ in range(shape_size(shape))
    ]
    for index, (left, right) in enumerate(
        zip(expanded_gradient._data, expanded_factor._data)
    ):
        coordinates = index_to_coordinates(index, expanded_gradient.shape)
        source_coordinates = tuple(
            0 if source_dimension == 1 else coordinate
            for source_dimension, coordinate in zip(padded_shape, coordinates)
        )[padding:]
        source_index = coordinates_to_index(source_coordinates, shape)
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


class ProductSumToShape:
    """Fused differentiable product reduction used by broadcast VJPs."""

    @staticmethod
    def forward(
        left: Tensor,
        right: Tensor,
        *,
        target_shape: tuple[int, ...],
    ) -> Tensor:
        return sum_products_to_shape(left, right, target_shape)

    @staticmethod
    def backward(
        grad: Tensor,
        *inputs: Tensor,
        **kwargs: object,
    ) -> list[Tensor]:
        from ..utils.broadcasting import broadcast_shape, broadcast_to

        left, right = inputs
        common_shape = broadcast_shape(left.shape, right.shape)
        expanded_grad = broadcast_to(grad, common_shape)
        return [
            sum_products_to_shape(expanded_grad, right, left.shape),
            sum_products_to_shape(expanded_grad, left, right.shape),
        ]

    @staticmethod
    def backward_graph(grad, *inputs, **kwargs: object):
        from ..utils.broadcasting import broadcast_shape

        left, right = inputs
        common_shape = broadcast_shape(left.shape, right.shape)
        ones = Tensor(
            [1.0] * shape_size(common_shape),
            dtype=grad.dtype,
            shape=common_shape,
        )
        expanded_grad = grad * ones
        return [
            sum_products_to_shape_graph(expanded_grad, right, left.shape),
            sum_products_to_shape_graph(expanded_grad, left, right.shape),
        ]


def sum_products_to_shape_graph(left, right, shape: tuple[int, ...]):
    """Record a fused multiply-and-reduce without premature overflow."""
    from ..variable import Variable

    return Variable._from_operation(
        ProductSumToShape.forward(left.data, right.data, target_shape=shape),
        "product_sum_to_shape",
        ProductSumToShape,
        [left, right],
        target_shape=shape,
    )


class ZeroLike:
    """A graph-connected zero that is safe for infinite input values."""

    @staticmethod
    def forward(value: Tensor) -> Tensor:
        return Tensor(
            [0.0] * value.size,
            dtype=value.dtype,
            shape=value.shape,
        )

    @staticmethod
    def backward(
        grad: Tensor,
        *inputs: Tensor,
        **kwargs: object,
    ) -> list[Tensor]:
        return [
            Tensor(
                [0.0] * inputs[0].size,
                dtype=grad.dtype,
                shape=inputs[0].shape,
            )
        ]

    @staticmethod
    def backward_graph(grad, *inputs, **kwargs: object):
        return [zero_like_graph(inputs[0])]


def zero_like_graph(value):
    """Return graph-connected zeros without evaluating ``value * 0``."""
    from ..variable import Variable

    return Variable._from_operation(
        ZeroLike.forward(value.data),
        "zero_like",
        ZeroLike,
        [value],
    )


class MaskedValue:
    """Select values with a constant mask without evaluating ``infinity * 0``."""

    @staticmethod
    def forward(value: Tensor, *, mask: Tensor) -> Tensor:
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

    @staticmethod
    def backward(
        grad: Tensor,
        *inputs: Tensor,
        **kwargs: object,
    ) -> list[Tensor]:
        mask = kwargs["mask"]
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

    @staticmethod
    def backward_graph(grad, *inputs, **kwargs: object):
        mask = kwargs["mask"]
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
    from ..utils.broadcasting import broadcast_shape
    from ..variable import Variable

    if broadcast_shape(value.shape, mask.shape) != mask.shape:
        raise ValueError(
            f"Value shape {value.shape} cannot broadcast to mask shape "
            f"{mask.shape}"
        )
    return Variable._from_operation(
        MaskedValue.forward(value.data, mask=mask),
        "masked_value",
        MaskedValue,
        [value],
        mask=mask,
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

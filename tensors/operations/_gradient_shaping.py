"""Gradient shaping that every differentiable domain needs.

A reverse pass rarely produces a gradient in the shape its operand wants. A
broadcast forward makes one value contribute to several output positions, so
its gradient is the sum of those contributions; a masked or short-circuited
forward makes some positions contribute nothing, so their gradient is a zero
of the right shape and dtype. Reshaping a gradient that way is part of a
derivative rule, not part of the operation being differentiated, and every
domain that broadcasts needs the same rules.

These live in the operation layer rather than one level down or one level up:

- They are not neutral utilities. ``ProductSumToShape``, ``ZeroLike``, and
  ``MaskedValue`` are :class:`~tensors.operations.base.Operation` subclasses
  with their own derivative rules, so a second-order pass can differentiate
  through them. :mod:`tensors.utils` holds primitives that know nothing about
  operations or gradients.
- They are not backend kernels. They decide *what* to compute for a VJP and
  hand the arithmetic to dispatch, exactly as any other operation does.

The module is private because these are not operations a user writes: they
appear only inside the ``backward`` and ``backward_graph`` of operations that
broadcast.
"""

from __future__ import annotations
from tensors.operations.base import Operation
from tensors.shape import Shape
from tensors.tensor import Tensor


def sum_to_shape(gradient, shape):
    """Reduce a broadcast gradient back to an operand's shape.

    A forward broadcast lets one operand value feed several output positions,
    so the gradient arrives at the output's shape and that value is owed the
    sum of the positions it fed. :meth:`~tensors.shape.Shape.stretched_axes_from`
    says which axes those are; summing them with ``keepdims`` leaves them in
    place, and dropping them afterwards is a relabelling rather than
    arithmetic.

    It is written with the sum and reshape operations, so the operands decide
    what the statements mean: given a Tensor they calculate, and given a
    Variable they record a reduction that can be differentiated again.

    The sum runs under the execution contract of `docs/backends.md`: no
    workload-size threshold decides where it happens, and a backend that
    cannot reduce conformingly reports that rather than letting the Python
    reference answer somewhere else.
    """
    from tensors.graph.expression import apply_operation, is_graph_operand
    from tensors.operations.manipulation.reshape import reshape
    from tensors.operations.reductions.sum import Sum

    shape = tuple(shape)
    if tuple(gradient.shape) == shape:
        return gradient
    axes = Shape.from_iterable(gradient.shape).stretched_axes_from(shape)
    reduction = Sum(axis=axes, keepdims=True, on_selected_backend=True)
    reduced = (
        apply_operation(reduction, (gradient,))
        if is_graph_operand(gradient)
        else reduction.forward(gradient)
    )
    return reduced if tuple(reduced.shape) == shape else reshape(reduced, shape)


class ProductSumToShape(Operation):
    """Fused differentiable product reduction used by broadcast VJPs."""

    __slots__ = ("target_shape",)
    name = "product_sum_to_shape"

    def __init__(self, *, target_shape: tuple[int, ...]) -> None:
        object.__setattr__(self, "target_shape", target_shape)

    def forward(self, left: Tensor, right: Tensor) -> Tensor:
        """Multiply and reduce in one step, so the product cannot lose range.

        Forming the products first can overflow to infinities that cancel to
        NaN, or underflow to zero, where the exact reduced result is
        representable. The kernel groups the factors before rounding them.
        """
        from tensors.backend import execute_sum_products_to_shape

        shape = self.target_shape
        accelerated = execute_sum_products_to_shape(left, right, shape)
        return Tensor._from_owned_storage(accelerated, dtype=left.dtype, shape=shape)

    def backward(self, grad, *inputs, needs_input_grad: tuple[bool, ...]):
        """Differentiate the fused product reduction, which is itself fused.

        Each operand's derivative is the same shape of computation as the
        multiplication VJP this operation implements: the upstream gradient
        times the other operand, grouped before rounding and summed back to
        that operand's shape. So it is expressed as this operation again,
        and the grouping the fused kernel performs is preserved at every
        derivative level rather than only at the first.

        The gradient arrives at the reduced shape and the reduction needs it
        at the shape the products had. Multiplying by ones is that expansion,
        exactly — ``x * 1`` is ``x`` for every value a gradient can hold —
        and it keeps the work on the selected backend, where the host-side
        broadcast it replaces did not.
        """
        from tensors.creation import ones
        from tensors.graph.expression import apply_operation, is_graph_operand

        left, right = inputs
        common_shape = left.shape.broadcast_with(right.shape)
        expanded = (
            grad
            if grad.shape == common_shape
            else grad * ones(common_shape, dtype=grad.dtype)
        )
        gradients = []
        for operand, factor, requested in (
            (left, right, needs_input_grad[0]),
            (right, left, needs_input_grad[1]),
        ):
            if not requested:
                gradients.append(None)
                continue
            reduction = ProductSumToShape(target_shape=operand.shape)
            gradients.append(
                apply_operation(reduction, (expanded, factor))
                if is_graph_operand(expanded)
                else reduction.forward(expanded, factor)
            )
        return gradients


class ZeroLike(Operation):
    """A graph-connected zero that is safe for infinite input values."""

    __slots__ = ()
    name = "zero_like"

    def forward(self, value: Tensor) -> Tensor:
        return Tensor([0.0] * value.size, dtype=value.dtype, shape=value.shape)

    def backward(
        self, grad: Tensor, *inputs: Tensor, needs_input_grad: tuple[bool, ...]
    ) -> list[Tensor]:
        return [Tensor([0.0] * inputs[0].size, dtype=grad.dtype, shape=inputs[0].shape)]

    def backward_graph(self, grad, *inputs, needs_input_grad: tuple[bool, ...]):
        return [zero_like_graph(inputs[0])]


def zero_like_graph(value):
    """Return graph-connected zeros without evaluating ``value * 0``."""
    from tensors.variable import Variable

    operation = ZeroLike()
    return Variable._apply_operation(operation, (value,))


class MaskedValue(Operation):
    """Select values with a constant mask without evaluating ``infinity * 0``."""

    __slots__ = ("mask",)
    name = "masked_value"

    def __init__(self, *, mask: Tensor) -> None:
        object.__setattr__(self, "mask", mask)

    def forward(self, value: Tensor) -> Tensor:
        mask = self.mask
        from tensors.utils.broadcasting import broadcast_to

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
        self, grad: Tensor, *inputs: Tensor, needs_input_grad: tuple[bool, ...]
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

    def backward_graph(self, grad, *inputs, needs_input_grad: tuple[bool, ...]):
        mask = self.mask
        if not isinstance(mask, Tensor):
            raise TypeError("masked-value mask must be a Tensor")
        return [sum_to_shape_graph(masked_value_graph(grad, mask), inputs[0].shape)]


def masked_value_graph(value, mask: Tensor):
    """Select graph values using a constant zero-one mask."""
    from tensors.variable import Variable

    if value.shape.broadcast_with(mask.shape) != mask.shape:
        raise ValueError(
            f"Value shape {value.shape} cannot broadcast to mask shape {mask.shape}"
        )
    operation = MaskedValue(mask=mask)
    return Variable._apply_operation(operation, (value,))


__all__ = [
    "MaskedValue",
    "ProductSumToShape",
    "masked_value_graph",
    "sum_to_shape",
    "sum_to_shape",
    "ZeroLike",
    "zero_like_graph",
]

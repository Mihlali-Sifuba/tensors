"""Shared gradient shaping that reverses the effect of a broadcast.

A reverse pass rarely produces a gradient in the shape its operand wants. A
forward broadcast lets one operand value feed several output positions, so the
gradient arrives at the output's shape and that value is owed the sum of the
positions it fed. Undoing that is part of a derivative rule rather than part of
the operation being differentiated, and every domain that broadcasts needs the
same rule, so it is stated once here.

This is not an :class:`~tensors.operations.base.Operation`, and deliberately.
An operation exists when a computation needs its own derivative rule; this one
composes ``sum`` and ``reshape``, which already have theirs, so it records and
differentiates through them without a vertex of its own. A computation with no
such composition — the fused product reduction in
:mod:`tensors.operations.reductions.product_sum_to_shape`, for instance —
is an operation for that reason, not because a VJP happens to call it.

Nothing here is a function a user writes. No facade re-exports it, so it is
internal whatever it is called; it appears only inside the ``backward`` of an
operation that broadcasts.
"""

from __future__ import annotations
from tensors.shape import Shape


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


__all__ = ["sum_to_shape"]

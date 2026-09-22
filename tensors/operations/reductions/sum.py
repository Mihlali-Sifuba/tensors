"""Sum and its differentiation rule."""

from __future__ import annotations
from tensors.backend import dispatch as backend_dispatch
from typing import TYPE_CHECKING, Any, List, overload
from tensors._typing import TensorData, TensorLike, TensorResult, TensorValue
from tensors.operations.base import Operation
from tensors.tensor import Tensor
from tensors.graph.expression import as_tensor_operand
from tensors.utils.reductions import (
    Axis,
    immutable_axis,
    keepdims_shape,
    normalize_axes,
    reduction_shape,
)

if TYPE_CHECKING:
    from tensors.graph.node import VariableNode


def _sum_impl(a: Tensor, axis: Axis = None, keepdims: bool = False) -> Tensor:
    """Sum over one, several, or all axes."""
    axes = normalize_axes(a.ndim, axis)
    output_shape = reduction_shape(a.shape, axes, keepdims)
    if axis is None and (not keepdims):
        output_shape = (1,)
    accelerated = backend_dispatch.execute_reduce_sum(
        a, axes, keepdims=keepdims, dtype=a.dtype, output_shape=output_shape
    )
    return Tensor._from_owned_storage(accelerated, dtype=a.dtype, shape=output_shape)


class Sum(Operation):
    """Sum with a reverse-mode gradient rule.

    ``on_selected_backend`` chooses where the reduction runs, not what it
    calculates. Off, the forward pass consults the workload-size policy and
    answers from the Python reference when the accelerated kernel declines,
    which is what ``ts.sum`` has always done. On, it runs under the execution
    contract of `docs/backends.md`: no policy, no reference, and a backend
    that cannot reduce conformingly reports that rather than letting another
    one answer. The addition VJP asks for it, because the broadcast gradient
    reduction is inside the arithmetic contract. It is the operation's own
    state, so a recorded reduction replays under the contract it was written
    with. It governs this forward reduction only; differentiating a sum is a
    different computation and keeps its own dispatch.
    """

    __slots__ = ("axis", "keepdims", "on_selected_backend")
    name = "sum"

    def __init__(
        self,
        *,
        axis: Axis = None,
        keepdims: bool = False,
        on_selected_backend: bool = False,
    ) -> None:
        object.__setattr__(self, "axis", axis)
        object.__setattr__(self, "keepdims", keepdims)
        object.__setattr__(self, "on_selected_backend", on_selected_backend)

    def forward(self, a: Tensor) -> Tensor:
        axis = self.axis
        keepdims = self.keepdims
        if not self.on_selected_backend:
            return _sum_impl(a, axis=axis, keepdims=keepdims)
        # The strict entry point names its reduction by the shape that keeps
        # the reduced axes rather than by the axes themselves, and the two say
        # the same thing. The collapsed shape is not one it could be given:
        # ``(2, 3)`` reduced over axis 1 collapses to ``(2,)``, which aligns
        # from the right against ``(2, 3)`` as a reduction it refuses. So the
        # reduction is asked for in the form that keeps the axes, and the flat
        # result it returns takes whichever shape ``keepdims`` asked for.
        axes = normalize_axes(a.ndim, axis)
        output_shape = reduction_shape(a.shape, axes, keepdims)
        if axis is None and (not keepdims):
            output_shape = (1,)
        accelerated = backend_dispatch.execute_vjp_sum_to_shape(
            a, reduction_shape(a.shape, axes, True)
        )
        return Tensor._from_owned_storage(
            accelerated, dtype=a.dtype, shape=output_shape
        )

    def backward(
        self, grad: Tensor, *inputs: Tensor, needs_input_grad: tuple[bool, ...]
    ) -> List[Tensor]:
        a = inputs[0]
        axis = self.axis
        keepdims = self.keepdims
        axes = normalize_axes(a.ndim, axis)
        output_shape = reduction_shape(a.shape, axes, keepdims)
        if axis is None and (not keepdims):
            output_shape = (1,)
        if grad.shape != output_shape:
            raise ValueError(
                f"Gradient shape {grad.shape} does not match output shape {output_shape}"
            )
        accelerated = backend_dispatch.execute_reduce_sum_gradient(
            grad, a, axes, keepdims=keepdims
        )
        return [
            Tensor._from_owned_storage(accelerated, dtype=grad.dtype, shape=a.shape)
        ]

    def backward_graph(self, grad, *inputs, needs_input_grad: tuple[bool, ...]):
        """Build a differentiable VJP for an axis-aware sum."""
        from tensors.creation import ones
        from tensors.variable import Variable
        from tensors.operations.manipulation.reshape import reshape

        axis = self.axis
        keepdims = self.keepdims
        value = inputs[0]
        expanded = (
            grad if keepdims else reshape(grad, keepdims_shape(value.shape, axis))
        )
        unit = Variable(ones(value.shape, dtype=grad.dtype), requires_grad=False)
        return [expanded * unit]


@overload
def sum(
    value: VariableNode, axis: Axis = None, keepdims: bool = False
) -> VariableNode: ...


@overload
def sum(
    value: TensorValue, axis: Axis = None, keepdims: bool = False
) -> TensorValue: ...


@overload
def sum(value: TensorData, axis: Axis = None, keepdims: bool = False) -> Tensor: ...


def sum(
    value: TensorLike | VariableNode, axis: Axis = None, keepdims: bool = False
) -> TensorResult | VariableNode:
    """Sum over one, several, or all axes.

    A graph value is applied through the graph: a Variable calculates the
    result now, and a vertex records the operation for a program that runs
    later. The configuration the operation is built with is the one every
    application uses, so a recorded call replays what it was written as.
    """
    from tensors.graph.expression import apply_operation, is_graph_operand

    axis = immutable_axis(axis)
    operation = Sum(axis=axis, keepdims=keepdims)
    if is_graph_operand(value):
        return apply_operation(operation, (value,))
    return operation.forward(as_tensor_operand(value))


__all__ = ["Sum", "sum"]

"""Reverse-mode automatic differentiation."""

from __future__ import annotations

from array import array
from typing import TYPE_CHECKING, Any

from ..graph.execution import topological_sort
from ..tensor import Tensor

if TYPE_CHECKING:
    from .variable import Variable


def backward(
    loss: Variable,
    grad: Tensor | array | list[Any] | int | float | None = None,
) -> None:
    """Differentiate ``loss`` with respect to all reachable Variables."""
    order = topological_sort([loss.node])

    for node in order:
        if node.output_var is not None:
            node.output_var.grad = None

    if grad is None:
        typecode = loss.dtype.typecode if loss.dtype.typecode in {"f", "d"} else "d"
        seed_data = array(typecode, [1.0] * loss.data.size)
        loss.grad = Tensor(seed_data, shape=loss.data.shape)
    else:
        seed = grad if isinstance(grad, Tensor) else Tensor(grad)
        if seed.shape != loss.data.shape:
            raise ValueError(
                f"Gradient shape {seed.shape} does not match loss shape {loss.data.shape}"
            )
        loss.grad = seed

    for node in reversed(order):
        if node.label == "var" or node.op_cls is None:
            continue

        output = node.output_var
        if output.grad is None:
            continue

        input_data = [edge.source.output_var.data for edge in node._in_edges]
        gradients = node.op_cls.backward(output.grad, *input_data, **node.args)
        for edge, gradient in zip(node._in_edges, gradients):
            _accumulate_gradient(edge.source.output_var, gradient)


def _accumulate_gradient(variable: Variable, gradient: Tensor) -> None:
    if not variable.requires_grad:
        return
    if variable.grad is None:
        variable.grad = gradient
        return

    from ..ops import Add

    variable.grad = Add.forward(variable.grad, gradient)


__all__ = ["backward"]

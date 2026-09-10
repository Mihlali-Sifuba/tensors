"""Elementwise hyperbolic cosine and its differentiation rule."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, overload

from .._typing import TensorData, TensorLike, TensorResult, TensorValue
from ..dtype import float64
from ..ops.operation import Operation
from ..tensor import Tensor
from ..graph.expression import as_tensor_operand
from ._unary import unary_backward, unary_forward

if TYPE_CHECKING:
    from ..graph.node import VariableNode


class Cosh(Operation):
    """Elementwise hyperbolic cosine with a reverse-mode gradient rule."""

    __slots__ = ()
    name = "cosh"

    def forward(self, value: Tensor) -> Tensor:
        dtype = value.dtype if value.dtype.typecode in {"f", "d"} else float64
        return unary_forward("cosh", value, dtype=dtype, fallback=_cosh)

    def backward(
        self,
        grad: Tensor,
        *inputs: Tensor,
        needs_input_grad: tuple[bool, ...],
    ) -> list[Tensor]:
        value = inputs[0]
        return [
            unary_backward(
                "cosh",
                grad,
                value,
                fallback=lambda upstream, item: upstream * _sinh(item),
            )
        ]

    def backward_graph(
        self,
        grad,
        *inputs,
        needs_input_grad: tuple[bool, ...],
    ):
        """Build a differentiable VJP for hyperbolic cosine."""
        from .sinh import sinh

        return [grad * sinh(inputs[0])]


@overload
def cosh(value: VariableNode) -> VariableNode: ...


@overload
def cosh(value: TensorValue) -> TensorValue: ...


@overload
def cosh(value: TensorData) -> Tensor: ...


def cosh(
    value: TensorLike | VariableNode,
) -> TensorResult | VariableNode:
    """Return the elementwise hyperbolic cosine.

    A graph value is applied through the graph: a Variable calculates the
    result now, and a vertex records the operation for a program that runs
    later.
    """
    from ..graph.expression import apply_operation, is_graph_operand

    if is_graph_operand(value):
        return apply_operation(Cosh(), (value,))
    value = as_tensor_operand(value)
    return Cosh().forward(value)


__all__ = ["Cosh", "cosh"]


def _cosh(value):
    try:
        return math.cosh(float(value))
    except OverflowError:
        return math.inf


def _sinh(value):
    try:
        return math.sinh(float(value))
    except OverflowError:
        return math.copysign(math.inf, value)

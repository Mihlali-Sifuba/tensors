"""Elementwise inverse hyperbolic tangent and its differentiation rule."""

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


class ArcTanh(Operation):
    """Elementwise inverse hyperbolic tangent on the real interval (-1, 1)."""

    __slots__ = ()
    name = "arctanh"

    def forward(self, value: Tensor) -> Tensor:
        dtype = value.dtype if value.dtype.typecode in {"f", "d"} else float64
        return unary_forward("arctanh", value, dtype=dtype, fallback=_arctanh)

    def backward(
        self,
        grad: Tensor,
        *inputs: Tensor,
        needs_input_grad: tuple[bool, ...],
    ) -> list[Tensor]:
        value = inputs[0]
        return [
            unary_backward(
                "arctanh",
                grad,
                value,
                fallback=lambda upstream, item: (
                    upstream / (1.0 - float(item) * float(item))
                ),
            )
        ]

    def backward_graph(
        self,
        grad,
        *inputs,
        needs_input_grad: tuple[bool, ...],
    ):
        """Build a differentiable VJP for inverse hyperbolic tangent."""
        value = inputs[0]
        return [grad / (1.0 - value ** 2.0)]


@overload
def arctanh(value: VariableNode) -> VariableNode: ...


@overload
def arctanh(value: TensorValue) -> TensorValue: ...


@overload
def arctanh(value: TensorData) -> Tensor: ...


def arctanh(
    value: TensorLike | VariableNode,
) -> TensorResult | VariableNode:
    """Return the elementwise inverse hyperbolic tangent.

    A graph value is applied through the graph: a Variable calculates the
    result now, and a vertex records the operation for a program that runs
    later.
    """
    from ..graph.expression import apply_operation, is_graph_operand

    if is_graph_operand(value):
        return apply_operation(ArcTanh(), (value,))
    value = as_tensor_operand(value)
    return ArcTanh().forward(value)


__all__ = ["ArcTanh", "arctanh"]


def _arctanh(value):
    value = float(value)
    if not math.isnan(value) and not -1.0 < value < 1.0:
        raise ValueError(
            "arctanh is only defined for values strictly between -1 and 1"
        )
    return math.atanh(value)

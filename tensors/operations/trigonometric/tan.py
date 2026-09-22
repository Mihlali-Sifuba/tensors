"""Elementwise tangent and its differentiation rule."""

from __future__ import annotations
from tensors.backend import dispatch as backend_dispatch
import math as _math
from typing import TYPE_CHECKING, Any, List, overload
from tensors._typing import TensorData, TensorLike, TensorResult, TensorValue
from tensors.dtype import float64
from tensors.operations.base import Operation
from tensors.tensor import Tensor
from tensors.graph.expression import as_tensor_operand

if TYPE_CHECKING:
    from tensors.graph.node import VariableNode


class Tan(Operation):
    """Elementwise tangent with a reverse-mode gradient rule."""

    __slots__ = ()
    name = "tan"

    def forward(self, a: Tensor) -> Tensor:
        dtype = a.dtype if a.dtype.typecode in {"f", "d"} else float64
        return Tensor._from_owned_storage(
            backend_dispatch.execute_tan(a, dtype=dtype), dtype=dtype, shape=a.shape
        )

    def backward(
        self, grad: Tensor, *inputs: Tensor, needs_input_grad: tuple[bool, ...]
    ) -> List[Tensor]:
        a = inputs[0]
        return [
            Tensor._from_owned_storage(
                backend_dispatch.execute_tan_gradient(grad, a),
                dtype=grad.dtype,
                shape=a.shape,
            )
        ]


@overload
def tan(value: VariableNode) -> VariableNode: ...


@overload
def tan(value: TensorValue) -> TensorValue: ...


@overload
def tan(value: TensorData) -> Tensor: ...


def tan(value: TensorLike | VariableNode) -> TensorResult | VariableNode:
    """Return the elementwise tangent of a graph value or Tensor.

    A graph value is applied through the graph: a Variable calculates the
    result now, and a vertex records the operation for a program that runs
    later.
    """
    from tensors.graph.expression import apply_operation, is_graph_operand

    if is_graph_operand(value):
        return apply_operation(Tan(), (value,))
    value = as_tensor_operand(value)
    return Tan().forward(value)


__all__ = ["Tan", "tan"]


def _tan_gradient(upstream, value):
    cosine = _math.cos(float(value))
    return upstream / (cosine * cosine)

"""Elementwise arctan and its differentiation rule."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, overload

from tensors._typing import TensorData, TensorLike, TensorResult, TensorValue
from tensors.backend import dispatch as backend_dispatch
from tensors.dtype import float64
from tensors.graph.expression import as_tensor_operand
from tensors.operations.base import Operation
from tensors.tensor import Tensor

if TYPE_CHECKING:
    from tensors.graph.node import VariableNode


class ArcTan(Operation):
    """Elementwise arctan with a reverse-mode gradient rule."""

    __slots__ = ()
    name = "arctan"

    def forward(self, value: Tensor) -> Tensor:
        dtype = value.dtype if value.dtype.typecode in {"f", "d"} else float64
        output_shape = value.shape
        return Tensor._from_owned_storage(
            backend_dispatch.execute_arctan(
                value, dtype=dtype, output_shape=output_shape
            ),
            dtype=dtype,
            shape=output_shape,
        )

    def backward(
        self, grad: Tensor, *inputs: Tensor, needs_input_grad: tuple[bool, ...]
    ) -> list[Optional[Tensor]]:
        """Apply the first-order VJP ``G / (1 + x**2)``."""
        if not needs_input_grad[0]:
            return [None]
        value = inputs[0]
        if grad.shape != value.shape:
            raise ValueError(
                f"Gradient shape {grad.shape} does not match value shape {value.shape}"
            )
        if grad.dtype is not value.dtype:
            raise ValueError(
                f"Gradient dtype {grad.dtype.name} does not match value dtype "
                f"{value.dtype.name}"
            )
        dtype = value.dtype
        output_shape = value.shape
        return [
            Tensor._from_owned_storage(
                backend_dispatch.execute_arctan_gradient(
                    grad, value, dtype=dtype, output_shape=output_shape
                ),
                dtype=dtype,
                shape=output_shape,
            )
        ]


@overload
def arctan(value: VariableNode) -> VariableNode: ...


@overload
def arctan(value: TensorValue) -> TensorValue: ...


@overload
def arctan(value: TensorData) -> Tensor: ...


def arctan(value: TensorLike | VariableNode) -> TensorResult | VariableNode:
    """Return the elementwise arctan of a graph value or Tensor."""
    from tensors.graph.expression import apply_operation, is_graph_operand

    if is_graph_operand(value):
        return apply_operation(ArcTan(), (value,))
    value = as_tensor_operand(value)
    return ArcTan().forward(value)


__all__ = ["ArcTan", "arctan"]

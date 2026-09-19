"""Broadcasting elementwise minimum."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, overload

from tensors._typing import TensorData, TensorLike, TensorResult
from tensors.backend import execute_minimum, execute_minimum_gradient
from tensors.dtype import result_dtype
from tensors.operations.selection._extremum import _ElementwiseExtremum, _extremum
from tensors.tensor import Tensor

if TYPE_CHECKING:
    from tensors.graph.node import VariableNode
    from tensors.variable import Variable


class Minimum(_ElementwiseExtremum):
    """Elementwise minimum with broadcasting."""

    __slots__ = ()
    name = "minimum"

    def forward(self, left: Tensor, right: Tensor) -> Tensor:
        """Select the smaller of each broadcast pair, propagating NaN."""
        shape = left.shape.broadcast_with(right.shape)
        dtype = result_dtype(left.dtype, right)
        storage = execute_minimum(left, right, dtype=dtype, output_shape=shape)
        return Tensor._from_owned_storage(storage, dtype=dtype, shape=shape)

    def backward(
        self, grad: Tensor, *inputs: Tensor, needs_input_grad: tuple[bool, ...]
    ) -> list[Optional[Tensor]]:
        """Route the upstream gradient to the selected operand."""
        left, right = inputs
        storages = execute_minimum_gradient(
            grad, left, right, needs_input_grad=needs_input_grad
        )
        return self._gradients(grad, left, right, storages)


@overload
def minimum(left: VariableNode, right: TensorLike | VariableNode) -> VariableNode: ...


@overload
def minimum(left: TensorLike, right: VariableNode) -> VariableNode: ...


@overload
def minimum(left: Variable, right: TensorLike) -> Variable: ...


@overload
def minimum(left: TensorLike, right: Variable) -> Variable: ...


@overload
def minimum(left: TensorData, right: TensorData) -> Tensor: ...


def minimum(
    left: TensorLike | VariableNode, right: TensorLike | VariableNode
) -> TensorResult | VariableNode:
    """Return the broadcasting elementwise minimum of two values."""
    return _extremum(Minimum(), left, right)


__all__ = ["Minimum", "minimum"]

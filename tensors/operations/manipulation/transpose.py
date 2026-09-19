"""Differentiable permutation of a tensor's axes."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, List, overload

from tensors._typing import TensorData, TensorLike, TensorResult, TensorValue
from tensors.operations.base import Operation
from tensors.tensor import Tensor
from tensors.graph.expression import as_tensor_operand
from tensors.backend import execute_transpose

if TYPE_CHECKING:
    from tensors.graph.node import VariableNode


def _transpose_impl(
    tensor: Tensor, axes: tuple[int, ...] | list[int] | None = None
) -> Tensor:
    """Permute tensor axes, defaulting to the final two matrix dimensions.

    This resolves the permutation and the output shape, then asks dispatch
    to produce the permuted storage. It is the transpose operation's own
    implementation helper, not a kernel: the numerical rearrangement is
    performed entirely by the selected backend.
    """
    if tensor.ndim < 2:
        raise ValueError("Transpose requires a tensor with at least 2D")
    if axes is None:
        permutation = tuple(range(tensor.ndim - 2)) + (tensor.ndim - 1, tensor.ndim - 2)
    else:
        permutation = tuple(axes)
        if any(
            (
                isinstance(axis, bool) or not isinstance(axis, int)
                for axis in permutation
            )
        ):
            raise TypeError("axes must contain only integers")
        normalized = tuple(
            (axis + tensor.ndim if axis < 0 else axis for axis in permutation)
        )
        if len(normalized) != tensor.ndim or set(normalized) != set(range(tensor.ndim)):
            raise ValueError(
                f"axes must be a permutation of 0..{tensor.ndim - 1}, got {permutation}"
            )
        permutation = normalized
    shape = tuple((tensor.shape[axis] for axis in permutation))
    accelerated = execute_transpose(tensor, permutation, output_shape=shape)
    return Tensor._from_owned_storage(accelerated, dtype=tensor.dtype, shape=shape)


class Transpose(Operation):
    """Transpose final matrix axes with a reverse-mode gradient rule."""

    __slots__ = ("axes",)
    name = "transpose"

    def __init__(self, *, axes: tuple[int, ...] | None = None) -> None:
        object.__setattr__(self, "axes", axes)

    def forward(self, value: Tensor) -> Tensor:
        """Permute the tensor axes described by this invocation."""
        return _transpose_impl(value, self.axes)

    def backward(
        self,
        grad: Tensor,
        *inputs: Tensor,
        needs_input_grad: tuple[bool, ...],
    ) -> List[Tensor]:
        """Transpose the upstream gradient back to the input layout."""
        axes = self.axes
        if axes is None:
            return [_transpose_impl(grad)]
        normalized = tuple(axis + grad.ndim if axis < 0 else axis for axis in axes)
        inverse = tuple(normalized.index(axis) for axis in range(grad.ndim))
        return [_transpose_impl(grad, inverse)]

    def backward_graph(
        self,
        grad,
        *inputs,
        needs_input_grad: tuple[bool, ...],
    ):
        """Build a differentiable VJP for transpose."""
        axes = self.axes
        if axes is None:
            return [transpose(grad)]
        normalized = tuple(axis + grad.ndim if axis < 0 else axis for axis in axes)
        inverse = tuple(normalized.index(axis) for axis in range(grad.ndim))
        return [transpose(grad, axes=inverse)]


@overload
def transpose(
    value: VariableNode,
    axes: tuple[int, ...] | list[int] | None = None,
) -> VariableNode: ...


@overload
def transpose(
    value: TensorValue,
    axes: tuple[int, ...] | list[int] | None = None,
) -> TensorValue: ...


@overload
def transpose(
    value: TensorData,
    axes: tuple[int, ...] | list[int] | None = None,
) -> Tensor: ...


def transpose(
    value: TensorLike | VariableNode,
    axes: tuple[int, ...] | list[int] | None = None,
) -> TensorResult | VariableNode:
    """Permute axes of a graph value or Tensor, or transpose the last two.

    A graph value is applied through the graph: a Variable calculates the
    result now, and a vertex records the operation for a program that runs
    later. The permutation is the operation's own state, so a recorded
    transpose replays the axes it was written with.
    """
    from tensors.graph.expression import apply_operation, is_graph_operand

    axes = tuple(axes) if isinstance(axes, list) else axes
    operation = Transpose(axes=axes)

    if is_graph_operand(value):
        return apply_operation(operation, (value,))
    return operation.forward(as_tensor_operand(value))


__all__ = ["Transpose", "transpose"]

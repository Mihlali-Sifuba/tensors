"""The matrix product over the final axes, with batch broadcasting."""

from __future__ import annotations
from typing import TYPE_CHECKING, List, Optional, overload
from tensors.backend import execute_matmul, execute_matmul_gradient
from tensors.backend.types import MatmulMetadata
from tensors._typing import TensorData, TensorLike, TensorResult
from tensors.dtype import result_dtype
from tensors.shape import Shape
from tensors.operations.base import Operation
from tensors.tensor import Tensor

if TYPE_CHECKING:
    from tensors.graph.node import VariableNode
    from tensors.variable import Variable


def _matmul_metadata(a: Tensor, b: Tensor) -> tuple[MatmulMetadata, tuple[int, ...]]:
    """Validate operands and return the shape information for matmul."""
    if a.ndim == 0 or b.ndim == 0:
        raise ValueError(
            "Matrix multiplication requires tensors with at least one dimension"
        )
    a_vector = a.ndim == 1
    b_vector = b.ndim == 1
    a_batch_shape = Shape() if a_vector else a.shape[:-2]
    b_batch_shape = Shape() if b_vector else b.shape[:-2]
    a_rows = 1 if a_vector else a.shape[-2]
    a_columns = a.shape[-1]
    b_rows = b.shape[0] if b_vector else b.shape[-2]
    b_columns = 1 if b_vector else b.shape[-1]
    if a_columns != b_rows:
        raise ValueError(
            f"Cannot multiply {a.shape} with {b.shape}: inner dimensions must match"
        )
    batch_shape = a_batch_shape.broadcast_with(b_batch_shape)
    if a_vector and b_vector:
        output_shape = ()
    elif a_vector:
        output_shape = batch_shape + (b_columns,)
    elif b_vector:
        output_shape = batch_shape + (a_rows,)
    else:
        output_shape = batch_shape + (a_rows, b_columns)
    return (
        (
            a_vector,
            b_vector,
            batch_shape,
            a_rows,
            a_columns,
            b_columns,
            a_batch_shape,
            b_batch_shape,
        ),
        output_shape,
    )


def _dot_impl(a: Tensor, b: Tensor) -> Tensor:
    """Return the NumPy-style matrix product of two tensors."""
    metadata, output_shape = _matmul_metadata(a, b)
    dtype = result_dtype(a.dtype, b)
    accelerated = execute_matmul(
        a, b, metadata=metadata, dtype=dtype, output_shape=output_shape
    )
    return Tensor._from_owned_storage(accelerated, dtype=dtype, shape=output_shape)


class MatMul(Operation):
    """The matrix product, with a reverse-mode gradient rule.

    ``ts.matmul`` and ``ts.dot`` are two names for this one operation:
    both contract the final axes with matrix-product semantics and
    broadcast any leading batch axes. They are not distinct contractions.
    """

    __slots__ = ()
    name = "matmul"

    def forward(self, a: Tensor, b: Tensor) -> Tensor:
        """Contract two tensors with matrix-product semantics."""
        return _dot_impl(a, b)

    def backward(
        self, grad: Tensor, *inputs: Tensor, needs_input_grad: tuple[bool, ...]
    ) -> List[Optional[Tensor]]:
        """Differentiate a matrix product with respect to requested operands."""
        a, b = inputs
        metadata, output_shape = _matmul_metadata(a, b)
        if grad.shape != output_shape:
            raise ValueError(
                f"Gradient shape {grad.shape} does not match output shape {output_shape}"
            )
        accelerated = execute_matmul_gradient(
            grad, a, b, metadata=metadata, needs_input_grad=needs_input_grad
        )
        a_storage, b_storage = accelerated
        return [
            (
                Tensor._from_owned_storage(a_storage, dtype=grad.dtype, shape=a.shape)
                if a_storage is not None
                else None
            ),
            (
                Tensor._from_owned_storage(b_storage, dtype=grad.dtype, shape=b.shape)
                if b_storage is not None
                else None
            ),
        ]


@overload
def matmul(a: VariableNode, b: TensorLike | VariableNode) -> VariableNode: ...


@overload
def matmul(a: TensorLike, b: VariableNode) -> VariableNode: ...


@overload
def matmul(a: Variable, b: TensorLike) -> Variable: ...


@overload
def matmul(a: TensorLike, b: Variable) -> Variable: ...


@overload
def matmul(a: TensorData, b: TensorData) -> Tensor: ...


def matmul(
    a: TensorLike | VariableNode, b: TensorLike | VariableNode
) -> TensorResult | VariableNode:
    """Return the general matrix product of two graph values or Tensors.

    A graph value on either side applies the product through the graph:
    Variables calculate it now, and a vertex records it for a program that
    runs later. A Tensor beside one enters the graph as a non-gradient leaf.
    """
    from tensors.graph.expression import (
        apply_operation,
        as_graph_operand,
        is_graph_operand,
    )

    if is_graph_operand(a) or is_graph_operand(b):
        return apply_operation(MatMul(), (as_graph_operand(a), as_graph_operand(b)))
    left = a if isinstance(a, Tensor) else Tensor(a)
    right = b if isinstance(b, Tensor) else Tensor(b)
    return MatMul().forward(left, right)


@overload
def dot(a: VariableNode, b: TensorLike | VariableNode) -> VariableNode: ...


@overload
def dot(a: TensorLike, b: VariableNode) -> VariableNode: ...


@overload
def dot(a: Variable, b: TensorLike) -> Variable: ...


@overload
def dot(a: TensorLike, b: Variable) -> Variable: ...


@overload
def dot(a: TensorData, b: TensorData) -> Tensor: ...


def dot(
    a: TensorLike | VariableNode, b: TensorLike | VariableNode
) -> TensorResult | VariableNode:
    """Return the general matrix product of two graph values or Tensors.

    The same operation as :func:`matmul`, under the name NumPy gives it. Both
    contract the final axes with matrix-product semantics and broadcast any
    leading batch axes; neither is a separate contraction.
    """
    return matmul(a, b)


__all__ = ["MatMul", "dot", "matmul"]

"""The matrix product over the final axes, with batch broadcasting."""

from __future__ import annotations
from typing import TYPE_CHECKING, Any, List, Optional, overload
from tensors.backend import execute_matmul, execute_matmul_gradient
from tensors._typing import TensorData, TensorLike, TensorResult
from tensors.dtype import result_dtype
from tensors.shape import Shape
from tensors.operations.base import Operation
from tensors.tensor import Tensor
from tensors.utils.coordinates import coordinates_to_linear_index

if TYPE_CHECKING:
    from tensors.graph.node import VariableNode
    from tensors.variable import Variable
MatmulMetadata = tuple[
    bool, bool, tuple[int, ...], int, int, int, tuple[int, ...], tuple[int, ...]
]


def _batch_coordinates(
    output_coordinates: tuple[int, ...], input_batch_shape: tuple[int, ...]
) -> tuple[int, ...]:
    """Map broadcasted batch coordinates to an input's batch coordinates."""
    padding = len(output_coordinates) - len(input_batch_shape)
    return tuple(
        (
            0 if dimension == 1 else output_coordinates[padding + index]
            for index, dimension in enumerate(input_batch_shape)
        )
    )


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


def _a_index(
    a: Tensor, a_vector: bool, batch_coordinates: tuple[int, ...], row: int, column: int
) -> int:
    """Return the logical linear index of an element in the left operand."""
    if a_vector:
        return column
    return coordinates_to_linear_index(batch_coordinates + (row, column), a.shape)


def _b_index(
    b: Tensor, b_vector: bool, batch_coordinates: tuple[int, ...], row: int, column: int
) -> int:
    """Return the logical linear index of an element in the right operand."""
    if b_vector:
        return row
    return coordinates_to_linear_index(batch_coordinates + (row, column), b.shape)


def _output_gradient(
    grad: Tensor,
    batch_coordinates: tuple[int, ...],
    row: int,
    column: int,
    a_vector: bool,
    b_vector: bool,
) -> Any:
    """Read an upstream gradient using the public matmul output shape."""
    if a_vector and b_vector:
        return grad._data[0]
    if a_vector:
        coordinates = batch_coordinates + (column,)
    elif b_vector:
        coordinates = batch_coordinates + (row,)
    else:
        coordinates = batch_coordinates + (row, column)
    return grad._data[coordinates_to_linear_index(coordinates, grad.shape)]


def _dot_impl(a: Tensor, b: Tensor) -> Tensor:
    """Return the NumPy-style matrix product of two tensors."""
    (
        a_vector,
        b_vector,
        batch_shape,
        a_rows,
        a_columns,
        b_columns,
        a_batch_shape,
        b_batch_shape,
    ), output_shape = _matmul_metadata(a, b)
    dtype = result_dtype(a.dtype, b)
    accelerated = execute_matmul(a, b, dtype=dtype, output_shape=output_shape)
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
        need_left, need_right = needs_input_grad
        (
            a_vector,
            b_vector,
            batch_shape,
            a_rows,
            a_columns,
            b_columns,
            a_batch_shape,
            b_batch_shape,
        ), output_shape = _matmul_metadata(a, b)
        if grad.shape != output_shape:
            raise ValueError(
                f"Gradient shape {grad.shape} does not match output shape {output_shape}"
            )
        accelerated = execute_matmul_gradient(
            grad, a, b, needs_input_grad=needs_input_grad
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

    def backward_graph(self, grad, *inputs, needs_input_grad: tuple[bool, ...]):
        """Build a differentiable VJP for vector and matrix products."""
        from tensors.operations.manipulation.transpose import transpose
        from tensors.operations.manipulation.reshape import reshape
        from tensors.operations._gradient_shaping import sum_to_shape_graph

        left, right = inputs
        need_left, need_right = needs_input_grad
        left_vector = left.ndim == 1
        right_vector = right.ndim == 1
        left_matrix = reshape(left, (1, left.shape[0])) if left_vector else left
        right_matrix = reshape(right, (right.shape[0], 1)) if right_vector else right
        if left_vector and right_vector:
            matrix_grad = reshape(grad, (1, 1))
        elif left_vector:
            matrix_grad = reshape(grad, grad.shape[:-1] + (1, grad.shape[-1]))
        elif right_vector:
            matrix_grad = reshape(grad, grad.shape + (1,))
        else:
            matrix_grad = grad
        left_gradient = None
        if need_left:
            left_gradient = sum_to_shape_graph(
                matrix_grad @ transpose(right_matrix), left_matrix.shape
            )
            if left_vector:
                left_gradient = reshape(left_gradient, left.shape)
        right_gradient = None
        if need_right:
            right_gradient = sum_to_shape_graph(
                transpose(left_matrix) @ matrix_grad, right_matrix.shape
            )
            if right_vector:
                right_gradient = reshape(right_gradient, right.shape)
        return [left_gradient, right_gradient]


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

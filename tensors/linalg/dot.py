"""General matrix multiplication and its differentiation rule."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, List, overload

from ..backend import (
    execute_matmul,
    execute_matmul_gradient,
    execute_transpose,
)
from .._typing import TensorData, TensorLike, TensorResult
from ..dtype import result_dtype
from ..math.sum import _stable_product_sum
from ..shape import Shape
from ..tensor import Tensor
from ..utils.coordinates import (
    coordinates_to_linear_index,
    linear_index_to_coordinates,
)

if TYPE_CHECKING:
    from ..variable import Variable


MatmulMetadata = tuple[
    bool,
    bool,
    tuple[int, ...],
    int,
    int,
    int,
    tuple[int, ...],
    tuple[int, ...],
]


def _batch_coordinates(
    output_coordinates: tuple[int, ...],
    input_batch_shape: tuple[int, ...],
) -> tuple[int, ...]:
    """Map broadcasted batch coordinates to an input's batch coordinates."""
    padding = len(output_coordinates) - len(input_batch_shape)
    return tuple(
        0 if dimension == 1 else output_coordinates[padding + index]
        for index, dimension in enumerate(input_batch_shape)
    )


def _matmul_metadata(
    a: Tensor,
    b: Tensor,
) -> tuple[MatmulMetadata, tuple[int, ...]]:
    """Validate operands and return the shape information for matmul."""
    if a.ndim == 0 or b.ndim == 0:
        raise ValueError("Matrix multiplication requires tensors with at least one dimension")

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
        a_vector,
        b_vector,
        batch_shape,
        a_rows,
        a_columns,
        b_columns,
        a_batch_shape,
        b_batch_shape,
    ), output_shape


def _a_index(
    a: Tensor,
    a_vector: bool,
    batch_coordinates: tuple[int, ...],
    row: int,
    column: int,
) -> int:
    """Return the flat index for an element in the left operand."""
    if a_vector:
        return column
    return coordinates_to_linear_index(
        batch_coordinates + (row, column),
        a.shape,
    )


def _b_index(
    b: Tensor,
    b_vector: bool,
    batch_coordinates: tuple[int, ...],
    row: int,
    column: int,
) -> int:
    """Return the flat index for an element in the right operand."""
    if b_vector:
        return row
    return coordinates_to_linear_index(
        batch_coordinates + (row, column),
        b.shape,
    )


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
    ) = _matmul_metadata(a, b)

    dtype = result_dtype(a.dtype, b)
    accelerated = execute_matmul(
        a,
        b,
        dtype=dtype,
        output_shape=output_shape,
    )
    if accelerated is not None:
        return Tensor(accelerated, dtype=dtype, shape=output_shape)

    values = []
    for batch_index in range(Shape.from_iterable(batch_shape).size):
        batch_coordinates = linear_index_to_coordinates(
            batch_index,
            batch_shape,
        )
        a_batch_coordinates = _batch_coordinates(batch_coordinates, a_batch_shape)
        b_batch_coordinates = _batch_coordinates(batch_coordinates, b_batch_shape)
        for row in range(a_rows):
            for column in range(b_columns):
                factors: list[tuple[int | float, int | float]] = []
                for inner in range(a_columns):
                    left = a._data[
                        _a_index(a, a_vector, a_batch_coordinates, row, inner)
                    ]
                    right = b._data[
                        _b_index(b, b_vector, b_batch_coordinates, inner, column)
                    ]
                    factors.append((left, right))
                if dtype.kind == "floating":
                    total = _stable_product_sum([
                        (float(left), float(right))
                        for left, right in factors
                    ])
                else:
                    total = sum(left * right for left, right in factors)
                values.append(total)

    return Tensor(values, dtype=dtype, shape=output_shape)


def _transpose_impl(
    tensor: Tensor,
    axes: tuple[int, ...] | list[int] | None = None,
) -> Tensor:
    """Permute tensor axes, defaulting to the final two matrix dimensions."""
    if tensor.ndim < 2:
        raise ValueError("Transpose requires a tensor with at least 2D")
    if axes is None:
        permutation = tuple(range(tensor.ndim - 2)) + (tensor.ndim - 1, tensor.ndim - 2)
    else:
        permutation = tuple(axes)
        if any(
            isinstance(axis, bool) or not isinstance(axis, int)
            for axis in permutation
        ):
            raise TypeError("axes must contain only integers")
        normalized = tuple(axis + tensor.ndim if axis < 0 else axis for axis in permutation)
        if len(normalized) != tensor.ndim or set(normalized) != set(range(tensor.ndim)):
            raise ValueError(
                f"axes must be a permutation of 0..{tensor.ndim - 1}, got {permutation}"
            )
        permutation = normalized

    shape = tuple(tensor.shape[axis] for axis in permutation)
    accelerated = execute_transpose(
        tensor,
        permutation,
        output_shape=shape,
    )
    if accelerated is not None:
        return Tensor(accelerated, dtype=tensor.dtype, shape=shape)
    inverse = [0] * tensor.ndim
    for output_axis, input_axis in enumerate(permutation):
        inverse[input_axis] = output_axis
    values = []
    for index in range(tensor.size):
        output_coordinates = linear_index_to_coordinates(index, shape)
        input_coordinates = tuple(
            output_coordinates[inverse[input_axis]]
            for input_axis in range(tensor.ndim)
        )
        values.append(
            tensor._data[
                coordinates_to_linear_index(input_coordinates, tensor.shape)
            ]
        )
    return Tensor(values, dtype=tensor.dtype, shape=shape)


class Dot:
    """General matrix multiplication with a reverse-mode gradient rule."""

    forward = staticmethod(_dot_impl)

    @staticmethod
    def backward(grad: Tensor, *inputs: Tensor, **kwargs: object) -> List[Tensor]:
        """Differentiate a matrix product with respect to both operands."""
        a, b = inputs
        (
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
        ) = _matmul_metadata(a, b)
        if grad.shape != output_shape:
            raise ValueError(
                f"Gradient shape {grad.shape} does not match output shape {output_shape}"
            )

        accelerated = execute_matmul_gradient(grad, a, b)
        if accelerated is not None:
            a_storage, b_storage = accelerated
            return [
                Tensor(a_storage, dtype=grad.dtype, shape=a.shape),
                Tensor(b_storage, dtype=grad.dtype, shape=b.shape),
            ]

        a_terms: list[list[tuple[float, float]]] = [[] for _ in range(a.size)]
        b_terms: list[list[tuple[float, float]]] = [[] for _ in range(b.size)]
        for batch_index in range(Shape.from_iterable(batch_shape).size):
            batch_coordinates = linear_index_to_coordinates(
                batch_index,
                batch_shape,
            )
            a_batch_coordinates = _batch_coordinates(batch_coordinates, a_batch_shape)
            b_batch_coordinates = _batch_coordinates(batch_coordinates, b_batch_shape)
            for row in range(a_rows):
                for column in range(b_columns):
                    upstream = _output_gradient(
                        grad,
                        batch_coordinates,
                        row,
                        column,
                        a_vector,
                        b_vector,
                    )
                    for inner in range(a_columns):
                        a_index = _a_index(a, a_vector, a_batch_coordinates, row, inner)
                        b_index = _b_index(b, b_vector, b_batch_coordinates, inner, column)
                        a_terms[a_index].append((
                            float(upstream),
                            float(b._data[b_index]),
                        ))
                        b_terms[b_index].append((
                            float(upstream),
                            float(a._data[a_index]),
                        ))

        a_values = [_stable_product_sum(terms) for terms in a_terms]
        b_values = [_stable_product_sum(terms) for terms in b_terms]

        return [
            Tensor(a_values, dtype=grad.dtype, shape=a.shape),
            Tensor(b_values, dtype=grad.dtype, shape=b.shape),
        ]

    @staticmethod
    def backward_graph(grad, *inputs, **kwargs: object):
        """Build a differentiable VJP for vector and matrix products."""
        from .transpose import transpose
        from ..math.reshape import reshape
        from ..ops._utils import sum_to_shape_graph

        left, right = inputs
        left_vector = left.ndim == 1
        right_vector = right.ndim == 1
        left_matrix = (
            reshape(left, (1, left.shape[0])) if left_vector else left
        )
        right_matrix = (
            reshape(right, (right.shape[0], 1)) if right_vector else right
        )

        if left_vector and right_vector:
            matrix_grad = reshape(grad, (1, 1))
        elif left_vector:
            matrix_grad = reshape(
                grad,
                grad.shape[:-1] + (1, grad.shape[-1]),
            )
        elif right_vector:
            matrix_grad = reshape(grad, grad.shape + (1,))
        else:
            matrix_grad = grad

        left_gradient = sum_to_shape_graph(
            matrix_grad @ transpose(right_matrix),
            left_matrix.shape,
        )
        right_gradient = sum_to_shape_graph(
            transpose(left_matrix) @ matrix_grad,
            right_matrix.shape,
        )
        if left_vector:
            left_gradient = reshape(left_gradient, left.shape)
        if right_vector:
            right_gradient = reshape(right_gradient, right.shape)
        return [left_gradient, right_gradient]


@overload
def dot(a: Variable, b: TensorLike) -> Variable: ...


@overload
def dot(a: TensorLike, b: Variable) -> Variable: ...


@overload
def dot(a: TensorData, b: TensorData) -> Tensor: ...


def dot(a: TensorLike, b: TensorLike) -> TensorResult:
    """Return the general matrix product of two Tensors or Variables."""
    from ..variable import Variable

    if isinstance(a, Variable) or isinstance(b, Variable):
        left = a if isinstance(a, Variable) else Variable(a, requires_grad=False)
        right = b if isinstance(b, Variable) else Variable(b, requires_grad=False)
        return Variable._from_operation(
            Dot.forward(left.data, right.data),
            "dot",
            Dot,
            [left, right],
        )

    left = a if isinstance(a, Tensor) else Tensor(a)
    right = b if isinstance(b, Tensor) else Tensor(b)
    return Dot.forward(left, right)


__all__ = ["Dot", "dot", "_transpose_impl"]

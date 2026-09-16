"""Reference the matrix product VJP for the Python backend."""

from __future__ import annotations
from tensors.backend.python.storage import PythonStorage
from typing import Any, TYPE_CHECKING
from tensors.math.sum import _stable_product_sum
from tensors.shape import Shape
from tensors.tensor import Tensor
from tensors.utils.coordinates import (
    coordinates_to_linear_index,
    linear_index_to_coordinates,
)

if TYPE_CHECKING:
    from tensors.backend.storage import Storage
MatmulMetadata = tuple[
    bool, bool, tuple[int, ...], int, int, int, tuple[int, ...], tuple[int, ...]
]


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


def matmul_gradient(
    grad: Tensor,
    left: Tensor,
    right: Tensor,
    *,
    needs_input_grad: tuple[bool, ...] = (True, True),
) -> tuple[Storage | None, Storage | None] | None:
    """Return the requested matrix-product VJPs."""
    a, b = (left, right)
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
    a_terms: list[list[tuple[float, float]]] = [
        [] for _ in range(a.size if need_left else 0)
    ]
    b_terms: list[list[tuple[float, float]]] = [
        [] for _ in range(b.size if need_right else 0)
    ]
    for batch_index in range(Shape.from_iterable(batch_shape).size):
        batch_coordinates = linear_index_to_coordinates(batch_index, batch_shape)
        a_batch_coordinates = _batch_coordinates(batch_coordinates, a_batch_shape)
        b_batch_coordinates = _batch_coordinates(batch_coordinates, b_batch_shape)
        for row in range(a_rows):
            for column in range(b_columns):
                upstream = _output_gradient(
                    grad, batch_coordinates, row, column, a_vector, b_vector
                )
                for inner in range(a_columns):
                    a_index = _a_index(a, a_vector, a_batch_coordinates, row, inner)
                    b_index = _b_index(b, b_vector, b_batch_coordinates, inner, column)
                    if need_left:
                        a_terms[a_index].append(
                            (float(upstream), float(b._data[b_index]))
                        )
                    if need_right:
                        b_terms[b_index].append(
                            (float(upstream), float(a._data[a_index]))
                        )
    return (
        (
            PythonStorage.from_values(
                [_stable_product_sum(terms) for terms in a_terms], grad.dtype
            )
            if need_left
            else None
        ),
        (
            PythonStorage.from_values(
                [_stable_product_sum(terms) for terms in b_terms], grad.dtype
            )
            if need_right
            else None
        ),
    )

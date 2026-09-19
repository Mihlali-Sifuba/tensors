"""Reference the matrix product for the Python backend."""

from __future__ import annotations
from tensors.backend.python.storage import PythonStorage
from tensors.backend.storage import Storage
from tensors.utils.summation import stable_product_sum
from tensors.shape import Shape
from tensors.tensor import Tensor
from tensors.utils.coordinates import (
    coordinates_to_linear_index,
    linear_index_to_coordinates,
)
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tensors.dtype import DataType
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


def matmul(
    left: Tensor, right: Tensor, *, dtype: DataType, output_shape: tuple[int, ...]
) -> Storage | None:
    """Accumulate each output element of the matrix product stably."""
    a = left
    b = right
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
    a_data = a._data
    b_data = b._data
    values = []
    if not batch_shape:
        for row in range(a_rows):
            a_row_offset = 0 if a_vector else row * a_columns
            for column in range(b_columns):
                factors = [
                    (
                        a_data[inner if a_vector else a_row_offset + inner],
                        b_data[inner if b_vector else inner * b_columns + column],
                    )
                    for inner in range(a_columns)
                ]
                if dtype.kind == "floating":
                    total = stable_product_sum(
                        [(float(left), float(right)) for left, right in factors]
                    )
                else:
                    total = sum((left * right for left, right in factors))
                values.append(total)
        return PythonStorage.from_values(values, dtype)
    for batch_index in range(Shape.from_iterable(batch_shape).size):
        batch_coordinates = linear_index_to_coordinates(batch_index, batch_shape)
        a_batch_coordinates = _batch_coordinates(batch_coordinates, a_batch_shape)
        b_batch_coordinates = _batch_coordinates(batch_coordinates, b_batch_shape)
        for row in range(a_rows):
            for column in range(b_columns):
                factors: list[tuple[int | float, int | float]] = []
                for inner in range(a_columns):
                    left = a_data[
                        _a_index(a, a_vector, a_batch_coordinates, row, inner)
                    ]
                    right = b_data[
                        _b_index(b, b_vector, b_batch_coordinates, inner, column)
                    ]
                    factors.append((left, right))
                if dtype.kind == "floating":
                    total = stable_product_sum(
                        [(float(left), float(right)) for left, right in factors]
                    )
                else:
                    total = sum((left * right for left, right in factors))
                values.append(total)
    return PythonStorage.from_values(values, dtype)

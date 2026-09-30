"""Reference matrix-product VJPs evaluated with Python arithmetic."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from tensors.backend.python.kernels.reductions.pairwise import pairwise_product_sum
from tensors.backend.python.kernels.reductions.sum_to_shape import sum_to_shape
from tensors.backend.python.storage import PythonStorage
from tensors.shape import Shape
from tensors.utils.coordinates import (
    coordinates_to_linear_index,
    linear_index_to_coordinates,
)

if TYPE_CHECKING:
    from tensors.backend.storage import Storage
    from tensors.dtype import DataType
    from tensors.backend.types import MatmulMetadata


def _batch_coordinates(
    output_coordinates: tuple[int, ...], input_batch_shape: tuple[int, ...]
) -> tuple[int, ...]:
    padding = len(output_coordinates) - len(input_batch_shape)
    return tuple(
        0 if dimension == 1 else output_coordinates[padding + index]
        for index, dimension in enumerate(input_batch_shape)
    )


def matmul_gradient(
    grad_values: Any,
    left_values: Any,
    right_values: Any,
    *,
    metadata: MatmulMetadata,
    left_shape: tuple[int, ...],
    right_shape: tuple[int, ...],
    dtype: DataType,
    needs_input_grad: tuple[bool, ...] = (True, True),
) -> tuple[Storage | None, Storage | None]:
    """Return the requested matrix-product VJPs.

    Each VJP is computed in two specified stages, as every backend computes
    it. First a contraction over the broadcast batch: the left VJP at
    ``(batch, row, inner)`` is the pairwise sum over ``column`` of
    ``grad * right``, and the right VJP at ``(batch, inner, column)`` is the
    pairwise sum over ``row`` of ``left * grad``, each product formed in the
    declared dtype. Then any batch axes the operand was broadcast along are
    summed back to its shape with the same pairwise tree.
    """
    (
        left_vector,
        right_vector,
        batch_shape,
        left_rows,
        inner_size,
        right_columns,
        left_batch_shape,
        right_batch_shape,
    ) = metadata
    need_left, need_right = needs_input_grad
    stored_left_shape = (
        (inner_size,) if left_vector else left_batch_shape + (left_rows, inner_size)
    )
    stored_right_shape = (
        (inner_size,)
        if right_vector
        else right_batch_shape + (inner_size, right_columns)
    )
    if left_vector and right_vector:
        grad_shape: tuple[int, ...] = ()
    elif left_vector:
        grad_shape = batch_shape + (right_columns,)
    elif right_vector:
        grad_shape = batch_shape + (left_rows,)
    else:
        grad_shape = batch_shape + (left_rows, right_columns)

    def upstream(batch: tuple[int, ...], row: int, column: int) -> Any:
        if left_vector and right_vector:
            return grad_values[0]
        if left_vector:
            coordinates = batch + (column,)
        elif right_vector:
            coordinates = batch + (row,)
        else:
            coordinates = batch + (row, column)
        return grad_values[coordinates_to_linear_index(coordinates, grad_shape)]

    def left_at(batch: tuple[int, ...], row: int, inner: int) -> Any:
        if left_vector:
            return left_values[inner]
        coordinates = _batch_coordinates(batch, left_batch_shape) + (row, inner)
        return left_values[coordinates_to_linear_index(coordinates, stored_left_shape)]

    def right_at(batch: tuple[int, ...], inner: int, column: int) -> Any:
        if right_vector:
            return right_values[inner]
        coordinates = _batch_coordinates(batch, right_batch_shape) + (inner, column)
        return right_values[
            coordinates_to_linear_index(coordinates, stored_right_shape)
        ]

    batches = [
        linear_index_to_coordinates(index, batch_shape)
        for index in range(Shape.from_iterable(batch_shape).size)
    ]
    rows = 1 if left_vector else left_rows
    columns = 1 if right_vector else right_columns

    left_storage = None
    if need_left:
        contracted = [
            pairwise_product_sum(
                [
                    (upstream(batch, row, column), right_at(batch, inner, column))
                    for column in range(columns)
                ],
                dtype,
            )
            for batch in batches
            for row in range(rows)
            for inner in range(inner_size)
        ]
        contracted_shape = batch_shape + (
            (inner_size,) if left_vector else (rows, inner_size)
        )
        left_storage = _reduced(contracted, contracted_shape, left_shape, dtype)

    right_storage = None
    if need_right:
        contracted = [
            pairwise_product_sum(
                [
                    (left_at(batch, row, inner), upstream(batch, row, column))
                    for row in range(rows)
                ],
                dtype,
            )
            for batch in batches
            for inner in range(inner_size)
            for column in range(columns)
        ]
        contracted_shape = batch_shape + (
            (inner_size,) if right_vector else (inner_size, columns)
        )
        right_storage = _reduced(contracted, contracted_shape, right_shape, dtype)
    return (left_storage, right_storage)


def _reduced(
    values: list[float],
    shape: tuple[int, ...],
    target: tuple[int, ...],
    dtype: DataType,
) -> Storage | None:
    """Sum a contracted VJP back over the batch axes its operand broadcast."""
    if tuple(shape) == tuple(target):
        return PythonStorage.from_values(values, dtype)
    return sum_to_shape(values, tuple(shape), tuple(target), dtype=dtype)

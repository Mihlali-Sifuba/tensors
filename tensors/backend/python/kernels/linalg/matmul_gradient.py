"""Reference matrix-product VJPs evaluated with Python arithmetic."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from tensors.backend.python.storage import PythonStorage
from tensors.shape import Shape
from tensors.utils.coordinates import (
    coordinates_to_linear_index,
    linear_index_to_coordinates,
)
from tensors.utils.summation import stable_product_sum

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
    """Return the requested matrix-product VJPs."""
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
    left_terms: list[list[tuple[float, float]]] = [
        [] for _ in range(Shape.from_iterable(left_shape).size if need_left else 0)
    ]
    right_terms: list[list[tuple[float, float]]] = [
        [] for _ in range(Shape.from_iterable(right_shape).size if need_right else 0)
    ]
    for batch_index in range(Shape.from_iterable(batch_shape).size):
        batch_coordinates = linear_index_to_coordinates(batch_index, batch_shape)
        left_batch = _batch_coordinates(batch_coordinates, left_batch_shape)
        right_batch = _batch_coordinates(batch_coordinates, right_batch_shape)
        for row in range(left_rows):
            for column in range(right_columns):
                if left_vector and right_vector:
                    upstream = grad_values[0]
                else:
                    output_coordinates = batch_coordinates
                    if left_vector:
                        output_coordinates += (column,)
                    elif right_vector:
                        output_coordinates += (row,)
                    else:
                        output_coordinates += (row, column)
                    output_shape = (
                        batch_shape + (right_columns,)
                        if left_vector
                        else (
                            batch_shape + (left_rows,)
                            if right_vector
                            else batch_shape + (left_rows, right_columns)
                        )
                    )
                    upstream = grad_values[
                        coordinates_to_linear_index(output_coordinates, output_shape)
                    ]
                for inner in range(inner_size):
                    left_index = (
                        inner
                        if left_vector
                        else coordinates_to_linear_index(
                            left_batch + (row, inner), left_shape
                        )
                    )
                    right_index = (
                        inner
                        if right_vector
                        else coordinates_to_linear_index(
                            right_batch + (inner, column), right_shape
                        )
                    )
                    if need_left:
                        left_terms[left_index].append(
                            (float(upstream), float(right_values[right_index]))
                        )
                    if need_right:
                        right_terms[right_index].append(
                            (float(upstream), float(left_values[left_index]))
                        )
    return (
        (
            PythonStorage.from_values(
                [stable_product_sum(terms) for terms in left_terms], dtype
            )
            if need_left
            else None
        ),
        (
            PythonStorage.from_values(
                [stable_product_sum(terms) for terms in right_terms], dtype
            )
            if need_right
            else None
        ),
    )

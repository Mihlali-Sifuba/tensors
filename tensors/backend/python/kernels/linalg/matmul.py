"""Reference matrix products evaluated with ordinary Python arithmetic."""

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


def matmul(
    left_values: Any,
    right_values: Any,
    *,
    metadata: MatmulMetadata,
    dtype: DataType,
    output_shape: tuple[int, ...],
) -> Storage:
    """Accumulate each output element according to matmul metadata."""
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
    left_shape = (
        (inner_size,) if left_vector else left_batch_shape + (left_rows, inner_size)
    )
    right_shape = (
        (inner_size,)
        if right_vector
        else right_batch_shape + (inner_size, right_columns)
    )
    values: list[int | float] = []
    for batch_index in range(Shape.from_iterable(batch_shape).size):
        batch_coordinates = linear_index_to_coordinates(batch_index, batch_shape)
        left_batch = _batch_coordinates(batch_coordinates, left_batch_shape)
        right_batch = _batch_coordinates(batch_coordinates, right_batch_shape)
        for row in range(left_rows):
            for column in range(right_columns):
                factors: list[tuple[int | float, int | float]] = []
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
                    factors.append((left_values[left_index], right_values[right_index]))
                if dtype.kind == "floating":
                    total = stable_product_sum(
                        [(float(left), float(right)) for left, right in factors]
                    )
                else:
                    total = sum(left * right for left, right in factors)
                values.append(total)
    return PythonStorage.from_values(values, dtype)

"""Apply SGD using Python arithmetic and its intermediate cast semantics."""

from itertools import repeat
from collections.abc import Iterable

from tensors.backend.python.kernels.arithmetic.multiply import multiply
from tensors.backend.python.kernels.arithmetic.subtract import subtract
from tensors.backend.storage import Storage
from tensors.dtype import DataType, result_dtype


def sgd_update(
    parameter_values: Iterable[int | float],
    gradient_values: Iterable[int | float],
    learning_rate: float,
    *,
    dtype: DataType,
    shape: tuple[int, ...],
) -> Storage:
    """Step each parameter against its gradient."""
    scaled_dtype = result_dtype(dtype, learning_rate)
    scaled_storage = multiply(
        gradient_values,
        repeat(learning_rate),
        dtype=scaled_dtype,
        output_shape=shape,
    )
    difference_dtype = result_dtype(dtype, scaled_storage)
    return subtract(
        parameter_values,
        scaled_storage.buffer,
        dtype=difference_dtype,
        output_shape=shape,
    )

"""Reference the product for the Python backend."""

from __future__ import annotations
from tensors.backend.python.storage import PythonStorage
from tensors.backend.storage import Storage
from tensors.dtype import DataType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tensors.tensor import Tensor
from tensors.utils.reductions import reduction_groups


def _product(values: list[int | float]) -> int | float:
    result: int | float = 1
    for value in values:
        result *= value
    return result


def reduce_prod(
    value: Tensor,
    axes: tuple[int, ...],
    *,
    keepdims: bool,
    dtype: DataType,
    output_shape: tuple[int, ...],
) -> Storage:
    """Return the product of each group."""
    _, output_shape, groups = reduction_groups(
        value.shape, axes, keepdims, scalar_as_vector=True
    )
    values = [_product([value._data[index] for index in group]) for group in groups]
    return PythonStorage.from_values(values, dtype)

"""Reference slicing for Python storage."""

from __future__ import annotations

from typing import TYPE_CHECKING

from tensors.backend.python.storage import PythonStorage
from tensors.backend.storage import Storage

if TYPE_CHECKING:
    from tensors._typing import TensorIndex
    from tensors.tensor import Tensor


def slice_tensor(
    value: Tensor,
    key: TensorIndex,
    *,
    output_shape: tuple[int, ...],
) -> Storage:
    """Copy the selected elements after caller-side key validation."""
    from tensors.utils.slicing import (
        slice_ranges_and_shape_from_key,
        storage_indices_from_ranges,
    )

    keys = key if isinstance(key, tuple) else (key,)
    ranges, _ = slice_ranges_and_shape_from_key(keys, value.shape)
    source = value._storage_for("python").buffer
    indices = storage_indices_from_ranges(
        ranges,
        value.shape,
        value.strides,
        value.offset,
    )
    return PythonStorage.from_values(
        (source[index] for index in indices),
        value.dtype,
    )

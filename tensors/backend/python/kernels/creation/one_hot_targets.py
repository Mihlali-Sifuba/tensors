"""Reference dense target expansion for the Python backend."""

from __future__ import annotations
from tensors.backend.python.storage import PythonStorage
from tensors.backend.storage import Storage
import math
from tensors.shape import Shape
from tensors.utils.coordinates import (
    coordinates_to_linear_index,
    linear_index_to_coordinates,
)
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tensors.tensor import Tensor


def one_hot_targets(logits: Tensor, targets: Tensor, axis: int) -> Storage | None:
    """Expand one validated class index per sample into a dense row."""
    sample_shape = logits.shape[:axis] + logits.shape[axis + 1 :]
    sample_count = Shape.from_iterable(sample_shape).size
    values = [0.0] * logits.size
    class_count = logits.shape[axis]
    for sample_index in range(sample_count):
        target = float(targets._data[sample_index])
        if not math.isfinite(target) or not target.is_integer():
            raise ValueError("Class-index targets must contain integers")
        class_index = int(target)
        if not 0 <= class_index < class_count:
            raise ValueError(f"Class index {class_index} is outside [0, {class_count})")
        sample_coordinates = linear_index_to_coordinates(sample_index, sample_shape)
        coordinates = (
            sample_coordinates[:axis] + (class_index,) + sample_coordinates[axis:]
        )
        values[coordinates_to_linear_index(coordinates, logits.shape)] = 1.0
    return PythonStorage.from_values(values, logits.dtype)

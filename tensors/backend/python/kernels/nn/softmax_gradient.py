"""Reference softmax vector-Jacobian product for the Python backend."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from tensors.backend.python.kernels.nn._normalization import (
    _axis_positions,
    _normalization_components,
)
from tensors.backend.python.storage import PythonStorage
from tensors.utils.summation import stable_product_sum

if TYPE_CHECKING:
    from tensors.dtype import DataType


def softmax_gradient(
    grad_values: Sequence[Any],
    value_values: Sequence[Any],
    grad_shape: tuple[int, ...],
    value_shape: tuple[int, ...],
    axis: int,
    *,
    dtype: DataType,
    value_dtype: DataType,
) -> PythonStorage | None:
    """Apply the softmax Jacobian without dominant cancellation."""
    if grad_shape != value_shape:
        return None
    probabilities, complements = _normalization_components(
        value_values, value_shape, axis, value_dtype
    )
    centered = [0.0] * math.prod(value_shape)
    for positions in _axis_positions(value_shape, axis):
        for position in positions:
            terms = [(float(grad_values[position]), complements[position])]
            terms.extend(
                (-float(grad_values[other]), probabilities[other])
                for other in positions
                if other != position
            )
            centered[position] = stable_product_sum(terms)
    rounded = PythonStorage.from_values(centered, dtype).buffer
    return PythonStorage.from_values(
        [
            probability * difference
            for probability, difference in zip(probabilities, rounded)
        ],
        dtype,
    )

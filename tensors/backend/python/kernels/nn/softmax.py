"""Reference softmax for the Python backend."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from tensors.backend.python.kernels.nn._normalization import _axis_positions
from tensors.backend.python.storage import PythonStorage
from tensors.utils.normalization import shifted_normalization

if TYPE_CHECKING:
    from tensors.dtype import DataType


def softmax(
    value_values: Sequence[Any],
    input_shape: tuple[int, ...],
    axis: int,
    *,
    dtype: DataType,
) -> PythonStorage:
    """Compute numerically stable softmax probabilities along ``axis``."""
    values = [0.0] * math.prod(input_shape)
    for positions in _axis_positions(input_shape, axis):
        group = [float(value_values[position]) for position in positions]
        if any(math.isnan(item) for item in group):
            for position in positions:
                values[position] = math.nan
            continue
        maximum = max(group)
        if maximum == math.inf:
            maxima = [
                position for position, item in zip(positions, group) if item == math.inf
            ]
            probability = 1.0 / len(maxima)
            for position in positions:
                values[position] = probability if position in maxima else 0.0
            continue
        if maximum == -math.inf:
            raise ValueError(
                "softmax is undefined when every value along an axis is -inf"
            )
        _, _, probabilities, _ = shifted_normalization(group)
        for position, probability in zip(positions, probabilities):
            values[position] = probability
    return PythonStorage.from_values(values, dtype)

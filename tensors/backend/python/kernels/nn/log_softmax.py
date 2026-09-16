"""Reference log-softmax for the Python backend."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from tensors.backend.python.kernels.nn._normalization import _axis_positions
from tensors.backend.python.storage import PythonStorage
from tensors.backend.storage import Storage
from tensors.math._normalization import shifted_normalization

if TYPE_CHECKING:
    from tensors.dtype import DataType
    from tensors.tensor import Tensor


def log_softmax(value: Tensor, axis: int, *, dtype: DataType) -> Storage:
    """Compute numerically stable log-softmax values along ``axis``."""
    values = [0.0] * value.size
    for positions in _axis_positions(value, axis):
        group = [float(value._data[position]) for position in positions]
        if any(math.isnan(item) for item in group):
            for position in positions:
                values[position] = math.nan
            continue
        maximum = max(group)
        if maximum == math.inf:
            maxima = [
                position for position, item in zip(positions, group) if item == math.inf
            ]
            selected = -math.log(len(maxima))
            for position in positions:
                values[position] = selected if position in maxima else -math.inf
            continue
        if maximum == -math.inf:
            raise ValueError(
                "log_softmax is undefined when every value along an axis is -inf"
            )
        _, correction, _, _ = shifted_normalization(group)
        for position, item in zip(positions, group):
            values[position] = item - maximum - correction
    return PythonStorage.from_values(values, dtype)

"""Reference the clipping VJP for the Python backend."""

from __future__ import annotations
import math
from tensors.backend.python.storage import PythonStorage
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tensors.backend.storage import Storage
    from tensors.tensor import Tensor


def clip_gradient(
    grad: Tensor,
    value: Tensor,
    min_value: int | float | None,
    max_value: int | float | None,
) -> Storage | None:
    """Pass the upstream gradient only where the value was unclipped."""
    mask = _mask(value, min_value, max_value)
    return PythonStorage.from_values(
        [upstream * weight for upstream, weight in zip(grad._data, mask)], grad.dtype
    )


def _mask(
    value: Tensor, min_value: int | float | None, max_value: int | float | None
) -> list[float]:
    mask = []
    for item in value._data:
        if isinstance(item, float) and math.isnan(item):
            mask.append(math.nan)
            continue
        above_minimum = min_value is None or item > min_value
        below_maximum = max_value is None or item < max_value
        mask.append(1.0 if above_minimum and below_maximum else 0.0)
    return mask

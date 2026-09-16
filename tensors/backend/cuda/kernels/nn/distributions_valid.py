"""CuPy implementation of the dense target predicate."""

from __future__ import annotations
import cupy
from typing import TYPE_CHECKING
from tensors.backend.cuda.conversion import _view

if TYPE_CHECKING:
    from tensors.tensor import Tensor


def distributions_valid(targets: Tensor, axis: int) -> bool:
    """Return whether dense targets are finite normalized probabilities."""
    values = _view(targets).astype(cupy.float64, copy=False)
    valid_values = cupy.all(cupy.isfinite(values) & (values >= 0.0) & (values <= 1.0))
    totals = cupy.sum(values, axis=axis)
    class_count = targets.shape[axis]
    epsilon = cupy.finfo(cupy.float64).eps
    accumulated_error = class_count * epsilon * cupy.sum(cupy.abs(values), axis=axis)
    tolerance = cupy.maximum(1e-07, 1e-07 * cupy.maximum(cupy.abs(totals), 1.0))
    valid_totals = cupy.all(cupy.abs(totals - 1.0) + accumulated_error <= tolerance)
    return bool(valid_values & valid_totals)

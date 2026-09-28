"""NumPy implementation of the dense target predicate."""

from __future__ import annotations
import numpy
from typing import TYPE_CHECKING
from tensors.backend.numpy.conversion import tensor_to_logical_array

if TYPE_CHECKING:
    from tensors.tensor import Tensor


def distributions_valid(targets: Tensor, axis: int) -> bool:
    """Return whether dense targets are finite normalized probabilities."""
    values = tensor_to_logical_array(targets).astype(numpy.float64, copy=False)
    valid_values = numpy.all(numpy.isfinite(values) & (values >= 0.0) & (values <= 1.0))
    totals = numpy.sum(values, axis=axis)
    class_count = targets.shape[axis]
    epsilon = numpy.finfo(numpy.float64).eps
    accumulated_error = class_count * epsilon * numpy.sum(numpy.abs(values), axis=axis)
    tolerance = numpy.maximum(1e-07, 1e-07 * numpy.maximum(numpy.abs(totals), 1.0))
    valid_totals = numpy.all(numpy.abs(totals - 1.0) + accumulated_error <= tolerance)
    return bool(valid_values & valid_totals)

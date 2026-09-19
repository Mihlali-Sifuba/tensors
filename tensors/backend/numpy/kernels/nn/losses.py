"""Cross-entropy and binary cross-entropy, and their VJPs."""

from __future__ import annotations
import numpy
from typing import Any
from typing import TYPE_CHECKING
from tensors.backend.numpy.conversion import _errstate

if TYPE_CHECKING:
    from tensors.backend.types import LossReduction


def _reduce_losses(values: Any, reduction: LossReduction) -> Any:
    """Reduce non-negative losses without overflowing an ordinary mean."""
    if reduction == "none":
        return values
    if reduction == "sum":
        with _errstate(over="ignore", invalid="ignore"):
            return numpy.asarray([numpy.sum(values)])
    scale = numpy.max(values)
    with _errstate(over="ignore", under="ignore", invalid="ignore"):
        exceptional = (scale == 0.0) | ~numpy.isfinite(scale)
        safe_scale = numpy.where(exceptional, 1.0, scale)
        stable = safe_scale * numpy.mean(values / safe_scale)
        result = numpy.where(exceptional, numpy.mean(values), stable)
    return numpy.asarray(result).reshape(1)

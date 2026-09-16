"""Cross-entropy and binary cross-entropy, and their VJPs."""

from __future__ import annotations
import cupy
from typing import Any
from typing import TYPE_CHECKING
from tensors.backend.cuda.conversion import _errstate

if TYPE_CHECKING:
    from tensors.backend.types import LossReduction


def _reduce_losses(values: Any, reduction: LossReduction) -> Any:
    """Reduce non-negative losses without overflowing an ordinary mean."""
    if reduction == "none":
        return values
    if reduction == "sum":
        with _errstate(over="ignore", invalid="ignore"):
            return cupy.asarray([cupy.sum(values)])
    scale = cupy.max(values)
    with _errstate(over="ignore", under="ignore", invalid="ignore"):
        exceptional = (scale == 0.0) | ~cupy.isfinite(scale)
        safe_scale = cupy.where(exceptional, 1.0, scale)
        stable = safe_scale * cupy.mean(values / safe_scale)
        result = cupy.where(exceptional, cupy.mean(values), stable)
    return cupy.asarray(result).reshape(1)

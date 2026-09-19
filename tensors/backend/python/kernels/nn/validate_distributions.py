"""Reference dense target validation for the Python backend."""

from __future__ import annotations
import math
from tensors.utils.reductions import reduction_groups
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tensors.tensor import Tensor


def validate_distributions(targets: Tensor, axis: int) -> None:
    """Raise unless every row along ``axis`` is a finite probability distribution."""
    _, _, groups = reduction_groups(targets.shape, axis, keepdims=False)
    for group in groups:
        values = [float(targets._data[index]) for index in group]
        if any(
            (not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in values)
        ):
            raise ValueError(
                "Dense cross-entropy targets must contain probabilities between 0 and 1"
            )
        if not math.isclose(math.fsum(values), 1.0, rel_tol=1e-07, abs_tol=1e-07):
            raise ValueError(
                "Dense cross-entropy targets must sum to 1 along the class axis"
            )
    return True

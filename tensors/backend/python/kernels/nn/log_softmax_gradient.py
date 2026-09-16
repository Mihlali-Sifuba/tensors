"""Reference log-softmax vector-Jacobian product for the Python backend."""

from __future__ import annotations

from typing import TYPE_CHECKING

from tensors.backend.python.kernels.nn._normalization import (
    _axis_positions,
    _normalization_components,
)
from tensors.backend.python.storage import PythonStorage
from tensors.backend.storage import Storage
from tensors.utils.summation import stable_product_sum

if TYPE_CHECKING:
    from tensors.tensor import Tensor


def log_softmax_gradient(grad: Tensor, value: Tensor, axis: int) -> Storage:
    """Apply the log-softmax Jacobian without dominant cancellation.

    Each row is ``g - p * sum(g)``, accumulated as one weighted sum in which
    the row's own term carries the complement of its probability, so a
    near-one probability keeps its significant digits.
    """
    probabilities, complements = _normalization_components(value, axis)
    values = [0.0] * value.size
    for positions in _axis_positions(value, axis):
        for position in positions:
            probability = probabilities[position]
            terms = [(float(grad._data[position]), complements[position])]
            terms.extend(
                (-float(grad._data[other]), probability)
                for other in positions
                if other != position
            )
            values[position] = stable_product_sum(terms)
    return PythonStorage.from_values(values, grad.dtype)

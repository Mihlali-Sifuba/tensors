"""Reference softmax vector-Jacobian product for the Python backend."""

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


def softmax_gradient(grad: Tensor, value: Tensor, axis: int) -> Storage:
    """Apply the softmax Jacobian without dominant cancellation.

    Each row is ``p * (g - E_p[g])``. The centered factor is accumulated from
    the complement of the row's own probability, so subtracting the expectation
    never loses the significant digits of a near-one probability. It is stored
    at the gradient's precision before scaling, as an intermediate tensor of
    that dtype would be.
    """
    probabilities, complements = _normalization_components(value, axis)
    centered = [0.0] * value.size
    for positions in _axis_positions(value, axis):
        for position in positions:
            terms = [(float(grad._data[position]), complements[position])]
            terms.extend(
                (-float(grad._data[other]), probabilities[other])
                for other in positions
                if other != position
            )
            centered[position] = stable_product_sum(terms)
    rounded = PythonStorage.from_values(centered, grad.dtype).buffer
    return PythonStorage.from_values(
        [
            probability * difference
            for probability, difference in zip(probabilities, rounded)
        ],
        grad.dtype,
    )

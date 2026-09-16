"""Loss functions comparing a prediction against a target."""

from tensors.operations.losses.binary_cross_entropy import (
    BinaryCrossEntropy,
    binary_cross_entropy,
)
from tensors.operations.losses.cross_entropy import (
    CrossEntropy,
    Reduction,
    cross_entropy,
)

__all__ = [
    "BinaryCrossEntropy",
    "binary_cross_entropy",
    "CrossEntropy",
    "Reduction",
    "cross_entropy",
]

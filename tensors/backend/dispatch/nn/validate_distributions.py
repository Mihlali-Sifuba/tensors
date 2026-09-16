"""Dispatch for normalization, probability, and loss kernels."""

from __future__ import annotations
from typing import TYPE_CHECKING
from tensors.backend.loading import _backend_kernel
from tensors.backend.policy import (
    _NUMPY_ELEMENTWISE_MIN_SIZE,
    _array_work_is_large_enough,
)

if TYPE_CHECKING:
    from tensors.tensor import Tensor


def execute_validate_distributions(targets: Tensor, axis: int) -> bool:
    """Validate dense probability rows with the active array backend."""
    from tensors.backend.python.kernels.nn.validate_distributions import (
        validate_distributions as reference,
    )

    if not _array_work_is_large_enough(targets.size, _NUMPY_ELEMENTWISE_MIN_SIZE):
        return reference(targets, axis)
    distributions_valid = _backend_kernel("distributions_valid")
    result = distributions_valid(targets, axis)
    if result is True:
        return result
    return reference(targets, axis)

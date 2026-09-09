"""Workload-size policy for accelerated execution.

These thresholds and predicates answer one question: is this operation big
enough to be worth handing to an array backend? Nothing here loads kernels
or executes work.
"""

from __future__ import annotations

from ..shape import Shape
from .config import get_backend


_NUMPY_ELEMENTWISE_MIN_SIZE = 32
_NUMPY_REDUCTION_MIN_SIZE = 8
_NUMPY_MATMUL_MIN_WORK = 32
_CUDA_FUSION_MIN_WORK = 8_192


def _shape_size(shape: tuple[int, ...]) -> int:
    return Shape.from_iterable(shape).size


def _array_work_is_large_enough(work: int, minimum: int) -> bool:
    backend = get_backend()
    if backend == "cuda":
        # Explicit CUDA selection keeps supported operations on-device. Kernel
        # implementations can still request the Python fallback when needed.
        return True
    return backend == "numpy" and work >= minimum

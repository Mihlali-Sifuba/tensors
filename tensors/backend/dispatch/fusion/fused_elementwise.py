"""Dispatch for fused elementwise chains.

The Python interpretation of a chain lives here with the dispatch entry point
that selects it, unchanged: it is the fallback the Python backend uses instead
of a compiled kernel.
"""

from __future__ import annotations
from collections.abc import Sequence
from typing import TYPE_CHECKING
from tensors.backend.config import get_backend
from tensors.backend.loading import _backend_kernel
from tensors.backend.policy import _CUDA_FUSION_MIN_WORK, _shape_size
from tensors.backend.storage import Storage
from tensors.backend.types import FusedElementwiseStep

if TYPE_CHECKING:
    from tensors.dtype import DataType
    from tensors.tensor import Tensor


def execute_fused_elementwise(
    values: Sequence[Tensor],
    steps: Sequence[FusedElementwiseStep],
    *,
    dtype: DataType,
    output_shape: tuple[int, ...],
) -> tuple[Storage, ...] | None:
    """Run a compatible floating-point expression chain as one backend plan."""
    from tensors.backend.python.kernels.fusion.fused_elementwise import (
        fused_elementwise as _execute_python_fused_elementwise,
    )

    work = _shape_size(output_shape)
    backend = get_backend()
    if not values or len(steps) < 2 or dtype.kind != "floating":
        return None
    if backend == "python":
        return _execute_python_fused_elementwise(
            values, steps, dtype=dtype, output_shape=output_shape
        )
    if backend == "cuda" and (
        work * len(steps) < _CUDA_FUSION_MIN_WORK and len(steps) < 64
    ):
        return None
    fused_elementwise = _backend_kernel("fused_elementwise")
    return fused_elementwise(
        tuple(values), tuple(steps), dtype=dtype, output_shape=output_shape
    )

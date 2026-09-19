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


def execute_fused_elementwise_backward(
    values: Sequence[Tensor],
    grad: Tensor,
    steps: Sequence[FusedElementwiseStep],
    *,
    dtype: DataType,
    output_shape: tuple[int, ...],
    requested_external: tuple[int, ...] = (),
) -> tuple[Storage, ...] | None:
    """Run the requested part of a chain VJP in one CUDA kernel.

    ``requested_external`` names the fused steps whose external operand
    gradient the current reverse pass wants. The backend refuses the request
    when its compact step form carries no such derivative, so the caller can
    fall back to ordinary operation execution.
    """
    work = _shape_size(output_shape)
    if (
        get_backend() != "cuda"
        or not values
        or len(steps) < 2
        or (dtype.kind != "floating")
        or (work * len(steps) < _CUDA_FUSION_MIN_WORK and len(steps) < 64)
    ):
        return None
    fused_backward = _backend_kernel("fused_elementwise_backward")
    return fused_backward(
        tuple(values),
        grad,
        tuple(steps),
        dtype=dtype,
        output_shape=output_shape,
        requested_external=requested_external,
    )

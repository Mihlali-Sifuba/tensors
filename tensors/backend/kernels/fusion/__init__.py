"""Fused-elementwise execution.

A chain of elementwise steps runs as one CUDA kernel (generated as source and
compiled through a cached RawKernel) or in one NumPy pass, forward and
backward."""

from __future__ import annotations

from .backward import (
    fused_elementwise_backward as fused_elementwise_backward,
)
from .forward import (
    fused_elementwise as fused_elementwise,
)


__all__ = [
    "fused_elementwise",
    "fused_elementwise_backward",
]

"""Kernels evaluated with CuPy device arrays."""

from tensors.backend.cuda.kernels.fusion.fused_elementwise import (
    fused_elementwise as fused_elementwise,
)
from tensors.backend.cuda.kernels.fusion.fused_elementwise_backward import (
    fused_elementwise_backward as fused_elementwise_backward,
)

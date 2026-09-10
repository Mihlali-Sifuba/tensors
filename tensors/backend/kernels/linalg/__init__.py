"""Linear-algebra kernels and their vector-Jacobian products."""

from __future__ import annotations

from .matmul_ops import (
    matmul as matmul,
    matmul_gradient as matmul_gradient,
)
from .outer_ops import (
    outer as outer,
    outer_gradient as outer_gradient,
)


__all__ = [
    "matmul",
    "matmul_gradient",
    "outer",
    "outer_gradient",
]

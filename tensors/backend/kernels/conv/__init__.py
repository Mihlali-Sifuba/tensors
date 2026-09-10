"""Grouped cross-correlation kernels and their vector-Jacobian products.

The package is named ``conv`` so that the exported ``convolution`` kernel does
not shadow a same-named module on this package."""

from __future__ import annotations

from .backward import (
    convolution_gradient as convolution_gradient,
)
from .forward import (
    convolution as convolution,
)


__all__ = [
    "convolution",
    "convolution_gradient",
]

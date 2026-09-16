"""Kernels evaluated with CuPy device arrays."""

from tensors.backend.cuda.kernels.convolution.convolution import (
    convolution as convolution,
)
from tensors.backend.cuda.kernels.convolution.convolution_gradient import (
    convolution_gradient as convolution_gradient,
)

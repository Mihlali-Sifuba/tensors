"""Kernels evaluated with NumPy arrays."""

from tensors.backend.numpy.kernels.linalg.matmul import matmul as matmul
from tensors.backend.numpy.kernels.linalg.matmul_gradient import (
    matmul_gradient as matmul_gradient,
)
from tensors.backend.numpy.kernels.linalg.outer import outer as outer
from tensors.backend.numpy.kernels.linalg.outer_gradient import (
    outer_gradient as outer_gradient,
)

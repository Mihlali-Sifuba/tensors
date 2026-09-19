"""Kernels evaluated with CuPy device arrays."""

from tensors.backend.cuda.kernels.linalg.matmul import matmul as matmul
from tensors.backend.cuda.kernels.linalg.matmul_gradient import (
    matmul_gradient as matmul_gradient,
)
from tensors.backend.cuda.kernels.linalg.outer import outer as outer
from tensors.backend.cuda.kernels.linalg.outer_gradient import (
    outer_gradient as outer_gradient,
)

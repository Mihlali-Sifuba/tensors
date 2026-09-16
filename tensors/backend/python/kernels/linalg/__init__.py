"""Reference kernels evaluated with ordinary Python arithmetic."""

from tensors.backend.python.kernels.linalg.matmul import matmul as matmul
from tensors.backend.python.kernels.linalg.matmul_gradient import (
    matmul_gradient as matmul_gradient,
)
from tensors.backend.python.kernels.linalg.outer import outer as outer
from tensors.backend.python.kernels.linalg.outer_gradient import (
    outer_gradient as outer_gradient,
)

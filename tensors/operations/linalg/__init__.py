"""Linear-algebra products."""

from tensors.operations.linalg.matmul import MatMul, dot, matmul
from tensors.operations.linalg.outer import Outer, outer

__all__ = [
    "MatMul",
    "dot",
    "matmul",
    "Outer",
    "outer",
]

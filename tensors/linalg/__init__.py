"""Linear-algebra operations."""

from .dot import Dot, dot
from .matmul import matmul
from .norm import Norm, norm
from .outer import Outer, outer
from .transpose import Transpose, transpose

__all__ = [
    "Dot", "dot", "matmul",
    "Norm", "norm",
    "Outer", "outer",
    "Transpose", "transpose",
]

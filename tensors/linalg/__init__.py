"""Linear-algebra operations."""

from .dot import Dot, dot
from .matmul import matmul
from .transpose import Transpose, transpose

__all__ = ["Dot", "dot", "matmul", "Transpose", "transpose"]

"""Linear-algebra operations."""

from .dot import Dot, dot
from .matmul import matmul
from .transpose import transpose

__all__ = ["Dot", "dot", "matmul", "transpose"]

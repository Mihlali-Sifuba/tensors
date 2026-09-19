"""Linear-algebra operations.

A convenience namespace over :mod:`tensors.operations`. ``matmul`` and
``outer`` are defined in :mod:`tensors.operations.linalg`; ``norm`` is a
reduction and is defined in :mod:`tensors.operations.reductions`, and
``transpose`` rearranges axes and is defined in
:mod:`tensors.operations.manipulation`. They are grouped here because that is
how a caller reaches for them.
"""

from tensors.operations.linalg import MatMul, dot, matmul, outer, Outer
from tensors.operations.manipulation import Transpose, transpose
from tensors.operations.reductions import Norm, norm

__all__ = [
    "MatMul",
    "Norm",
    "Outer",
    "Transpose",
    "dot",
    "matmul",
    "norm",
    "outer",
    "transpose",
]

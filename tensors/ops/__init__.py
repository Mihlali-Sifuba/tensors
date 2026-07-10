"""Operations — each op is a class with forward/backward for autograd.

Individual op classes:
    Add, Sub, Mul, Div, Neg, Dot, Sum, Mean, Slice

The ``Ops`` namespace provides the old interface for direct use::

    from tensors import Ops
    result = Ops.add(a, b)          # calls Add.forward(a, b)
"""

from __future__ import annotations

import builtins
import math
from typing import Tuple, Union

from .add import Add
from .sub import Sub
from .mul import Mul
from .div import Div
from .neg import Neg
from .dot import Dot, _transpose_impl
from .sum import Sum
from .mean import Mean
from .slice import Slice


from ..tensor import Tensor


class Ops:
    """Operation namespace — mirrors the old static-method interface.

    Each arithmetic/delegate method forwards to the corresponding
    op class's forward method. Non-differentiable utilities (transpose,
    reshape, min, max, std) are defined directly here.
    """

    # -- Arithmetic (delegates to op class forward) --------------------

    add = staticmethod(Add.forward)
    subtract = staticmethod(Sub.forward)
    multiply = staticmethod(Mul.forward)
    divide = staticmethod(Div.forward)
    dot = staticmethod(Dot.forward)
    neg = staticmethod(Neg.forward)
    sum_op = staticmethod(Sum.forward)
    mean_op = staticmethod(Mean.forward)

    # -- Shape utilities -----------------------------------------------

    @staticmethod
    def transpose(tensor: Tensor) -> Tensor:
        """Transpose a 2D tensor (swap rows and columns)."""
        return _transpose_impl(tensor)

    @staticmethod
    def reshape(tensor: Tensor, *new_shape: int) -> Tensor:
        """Reshape a tensor (total elements must match)."""
        from ..tensor import Tensor as _Tensor
        total = tensor._get_total_elements()
        new_total = 1
        for dim in new_shape:
            new_total *= dim
        if total != new_total:
            raise ValueError(
                f"Cannot reshape tensor of size {total} to shape {new_shape}"
            )
        return _Tensor(tensor._data, shape=new_shape)

    # -- Statistics (non-differentiable) -------------------------------

    @staticmethod
    def sum(tensor: Tensor) -> float:
        """Sum of all elements (returns float)."""
        return builtins.sum(tensor._data)

    @staticmethod
    def mean(tensor: Tensor) -> float:
        """Mean of all elements (returns float)."""
        if tensor.size == 0:
            return 0.0
        return builtins.sum(tensor._data) / tensor.size

    @staticmethod
    def min(tensor: Tensor) -> float:
        """Minimum value in the tensor."""
        if tensor.size == 0:
            raise ValueError("Cannot compute min of empty tensor")
        return builtins.min(tensor._data)

    @staticmethod
    def max(tensor: Tensor) -> float:
        """Maximum value in the tensor."""
        if tensor.size == 0:
            raise ValueError("Cannot compute max of empty tensor")
        return builtins.max(tensor._data)

    @staticmethod
    def std(tensor: Tensor) -> float:
        """Standard deviation of all elements (returns float)."""
        if tensor.size == 0:
            return 0.0
        m = Ops.mean(tensor)
        variance = builtins.sum((x - m) ** 2 for x in tensor._data) / tensor.size
        return math.sqrt(variance)


__all__ = [
    "Add", "Sub", "Mul", "Div", "Neg", "Dot", "Sum", "Mean", "Slice",
    "Ops",
]

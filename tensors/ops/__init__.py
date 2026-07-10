"""Operations — each op is a class with forward/backward differentiation rules.

Individual op classes:
    Add, Sub, Mul, Div, Neg, Slice

The ``Ops`` namespace provides the old interface for direct use::

    from tensors import Ops
    result = Ops.add(a, b)          # calls Add.forward(a, b)
"""

from __future__ import annotations

from typing import Tuple, Union

from .add import Add
from .sub import Sub
from .mul import Mul
from .div import Div
from .neg import Neg
from .slice import Slice


from ..tensor import Tensor


class Ops:
    """Operation namespace — mirrors the old static-method interface.

    Each arithmetic/delegate method forwards to the corresponding op class's
    forward method. Reshape remains here temporarily as a tensor-structure
    compatibility helper.
    """

    # -- Arithmetic (delegates to op class forward) --------------------

    add = staticmethod(Add.forward)
    subtract = staticmethod(Sub.forward)
    multiply = staticmethod(Mul.forward)
    divide = staticmethod(Div.forward)
    neg = staticmethod(Neg.forward)

    # -- Shape utilities -----------------------------------------------

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

__all__ = [
    "Add", "Sub", "Mul", "Div", "Neg", "Slice",
    "Ops",
]

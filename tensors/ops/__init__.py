"""Operations — each op is a class with forward/backward differentiation rules.

Individual op classes:
    Add, Sub, Mul, Div, Neg, Slice

The ``Ops`` namespace provides the old interface for direct use::

    from tensors import Ops
    result = Ops.add(a, b)          # evaluates Add().forward(a, b)
"""

from __future__ import annotations

from typing import Tuple, Union

from .add import Add, add
from .sub import Sub, subtract
from .mul import Mul, multiply
from .div import Div, divide, divide_scalar
from .neg import Neg, negate
from .slice import Slice
from .pow import Pow, pow, power, power_scalar_base
from .cast import Cast


from ..tensor import Tensor


class Ops:
    """Operation namespace — mirrors the old static-method interface.

    Each arithmetic/delegate method forwards to the corresponding op class's
    forward method. Reshape remains here temporarily as a tensor-structure
    compatibility helper.
    """

    # -- Arithmetic (delegates to a configuration-free invocation) -----

    add = staticmethod(add)
    subtract = staticmethod(subtract)
    multiply = staticmethod(multiply)
    divide = staticmethod(divide)
    pow = staticmethod(power)
    neg = staticmethod(negate)

__all__ = [
    "add",
    "subtract",
    "multiply",
    "divide",
    "divide_scalar",
    "negate",
    "power",
    "power_scalar_base",
    "Add",
    "Sub",
    "Mul",
    "Div",
    "Pow",
    "Neg",
    "Slice",
    "Cast",
    "pow",
    "Ops",
]

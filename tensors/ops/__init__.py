"""The operation contract and the primitive arithmetic operations.

A convenience namespace over :mod:`tensors.operations`, kept because
:class:`Operation` is the documented extension point for a custom operation.
The operations themselves are defined in :mod:`tensors.operations`.
"""

from tensors.operations.arithmetic import (
    Add,
    Div,
    Mul,
    Neg,
    Pow,
    Sub,
    add,
    divide,
    divide_scalar,
    multiply,
    negate,
    pow,
    power,
    power_scalar_base,
    subtract,
)
from tensors.operations.base import Operation
from tensors.operations.manipulation import Cast, Slice

__all__ = [
    "Add",
    "Cast",
    "Div",
    "Mul",
    "Neg",
    "Operation",
    "Pow",
    "Slice",
    "Sub",
    "add",
    "divide",
    "divide_scalar",
    "multiply",
    "negate",
    "pow",
    "power",
    "power_scalar_base",
    "subtract",
]

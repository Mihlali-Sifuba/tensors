"""Automatic differentiation for Variables."""

from .variable import Variable, dot, sum, mean
from .backward import backward

__all__ = [
    "Variable",
    "backward",
    "dot",
    "sum",
    "mean",
]

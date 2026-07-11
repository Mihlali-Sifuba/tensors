"""Mathematical functions for tensors and differentiable Variables."""

from .sum import Sum, sum
from .mean import Mean, mean
from .min import min
from .max import max
from .std import std
from .reshape import reshape

__all__ = ["Sum", "Mean", "sum", "mean", "min", "max", "std", "reshape"]

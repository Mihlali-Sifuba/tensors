"""Operations that collapse one or more axes."""

from tensors.operations.reductions.argmax import ArgMax, argmax
from tensors.operations.reductions.argmin import ArgMin, argmin
from tensors.operations.reductions.logsumexp import LogSumExp, logsumexp
from tensors.operations.reductions.max import Max, max
from tensors.operations.reductions.mean import Mean, mean
from tensors.operations.reductions.min import Min, min
from tensors.operations.reductions.norm import Norm, norm
from tensors.operations.reductions.prod import Prod, prod
from tensors.operations.reductions.std import Std, std
from tensors.operations.reductions.sum import Sum, sum
from tensors.operations.reductions.variance import Variance, variance

__all__ = [
    "ArgMax",
    "argmax",
    "ArgMin",
    "argmin",
    "LogSumExp",
    "logsumexp",
    "Max",
    "max",
    "Mean",
    "mean",
    "Min",
    "min",
    "Norm",
    "norm",
    "Prod",
    "prod",
    "Std",
    "std",
    "Sum",
    "sum",
    "Variance",
    "variance",
]

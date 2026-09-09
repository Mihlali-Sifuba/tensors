"""Type aliases naming the backends and the operations they implement."""

from __future__ import annotations

from typing import Literal, TypeAlias


BackendName: TypeAlias = Literal["python", "numpy", "cuda"]
BackendSelection: TypeAlias = Literal["python", "numpy", "cuda", "auto"]
BinaryOperation: TypeAlias = Literal[
    "add",
    "subtract",
    "multiply",
    "divide",
    "power",
]
ReductionOperation: TypeAlias = Literal[
    "sum",
    "mean",
    "variance",
    "std",
    "prod",
    "min",
    "max",
    "norm",
]
DifferentiableReductionOperation: TypeAlias = Literal[
    "sum",
    "mean",
    "variance",
    "std",
    "prod",
    "min",
    "max",
]
ComparisonOperation: TypeAlias = Literal[
    "equal",
    "not_equal",
    "less",
    "less_equal",
    "greater",
    "greater_equal",
]
ExtremumOperation: TypeAlias = Literal["minimum", "maximum"]
ArgExtremumOperation: TypeAlias = Literal["argmin", "argmax"]
NormalizationOperation: TypeAlias = Literal[
    "softmax",
    "log_softmax",
]
LossReduction: TypeAlias = Literal["none", "mean", "sum"]
UnaryOperation: TypeAlias = Literal[
    "abs",
    "sqrt",
    "exp",
    "log",
    "sin",
    "cos",
    "tan",
    "arcsin",
    "arccos",
    "arctan",
    "sinh",
    "cosh",
    "arcsinh",
    "arccosh",
    "arctanh",
    "sign",
    "relu",
    "sigmoid",
    "tanh",
    "softplus",
]
FusedElementwiseStep: TypeAlias = tuple[
    str,
    float | None,
    bool,
    int | None,
]

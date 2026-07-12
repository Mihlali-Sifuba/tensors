from .tensor import Tensor
from . import graph, linalg, math, ops, optim
from .ops import Ops, pow
from .dtype import DataType, float64, float32, int64, int32, int16, int8, uint8
from .variable import Variable
from .graph import Graph, backward
from .linalg import dot, matmul, transpose
from .math import (
    sum, mean, min, max,
    sqrt, exp, log, relu, sigmoid, tanh, softplus,
    std, reshape, stack, concat,
)

# Lift all Ops static methods to package-level functions
add = Ops.add
subtract = Ops.subtract
multiply = Ops.multiply
divide = Ops.divide

__all__ = [
    "Tensor", "Variable", "Graph", "backward", "Ops", "DataType",
    "graph", "ops", "linalg", "math", "optim",
    "float64", "float32", "int64", "int32", "int16", "int8", "uint8",
    "add", "subtract", "multiply", "divide", "pow",
    "dot", "matmul", "transpose", "reshape", "stack", "concat",
    "sum", "mean", "min", "max",
    "sqrt", "exp", "log", "relu", "sigmoid", "tanh", "softplus", "std",
]

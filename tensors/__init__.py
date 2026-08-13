from .tensor import Tensor
from . import graph, linalg, math, ops, optim
from .ops import Ops, pow
from .dtype import DataType, float64, float32, int64, int32, int16, int8, uint8
from .variable import Variable
from .graph import Graph, backward, grad
from .linalg import dot, matmul, norm, outer, transpose
from .math import (
    sum, mean, min, max,
    sqrt, exp, log, relu, sigmoid, tanh, softplus, softmax,
    std, reshape, stack, concat,
)

# Lift all Ops static methods to package-level functions
add = Ops.add
subtract = Ops.subtract
multiply = Ops.multiply
divide = Ops.divide

__all__ = [
    "Tensor", "Variable", "Graph", "backward", "grad", "Ops", "DataType",
    "graph", "ops", "linalg", "math", "optim",
    "float64", "float32", "int64", "int32", "int16", "int8", "uint8",
    "add", "subtract", "multiply", "divide", "pow",
    "dot", "matmul", "norm", "outer", "transpose", "reshape", "stack", "concat",
    "sum", "mean", "min", "max",
    "sqrt", "exp", "log", "relu", "sigmoid", "tanh", "softplus", "softmax", "std",
]

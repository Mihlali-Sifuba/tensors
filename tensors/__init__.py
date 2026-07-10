from .tensor import Tensor
from . import graph, linalg, math, ops, optim
from .ops import Ops
from .dtype import DataType, float64, float32, int64, int32, int16, int8, uint8
from .variable import Variable
from .graph import Graph, backward
from .linalg import dot, transpose
from .math import sum, mean, min, max, std

# Lift all Ops static methods to package-level functions
add = Ops.add
subtract = Ops.subtract
multiply = Ops.multiply
divide = Ops.divide
reshape = Ops.reshape

__all__ = [
    "Tensor", "Variable", "Graph", "backward", "Ops", "DataType",
    "graph", "ops", "linalg", "math", "optim",
    "float64", "float32", "int64", "int32", "int16", "int8", "uint8",
    "add", "subtract", "multiply", "divide",
    "dot", "transpose", "reshape", "sum", "mean", "min", "max", "std",
]

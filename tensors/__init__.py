from .tensor import Tensor
from .ops import Ops
from .dtype import DataType, float64, float32, int64, int32, int16, int8, uint8

# Lift all Ops static methods to package-level functions
add = Ops.add
subtract = Ops.subtract
multiply = Ops.multiply
divide = Ops.divide
dot = Ops.dot
transpose = Ops.transpose
reshape = Ops.reshape
sum = Ops.sum
mean = Ops.mean
min = Ops.min
max = Ops.max
std = Ops.std

__all__ = [
    "Tensor", "Ops", "DataType",
    "float64", "float32", "int64", "int32", "int16", "int8", "uint8",
    "add", "subtract", "multiply", "divide",
    "dot", "transpose", "reshape", "sum", "mean", "min", "max", "std",
]
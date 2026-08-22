from .tensor import Tensor
from . import creation, graph, linalg, math, ops, optim
from .creation import arange, eye, full, linspace, ones, zeros
from .ops import Ops, pow
from .dtype import DataType, float64, float32, int64, int32, int16, int8, uint8
from .variable import Variable
from .graph import (
    GradcheckError, Graph, backward, grad, gradcheck, hessian, jacobian,
)
from .linalg import dot, matmul, norm, outer, transpose
from .math import (
    sum, mean, min, max,
    sqrt, exp, log, sin, cos, tan, arcsin, arccos, arctan,
    sinh, cosh, arcsinh, arccosh, arctanh, sign,
    relu, sigmoid, tanh, softplus, softmax,
    logsumexp, log_softmax, cross_entropy, binary_cross_entropy,
    std, variance, reshape, stack, concat, abs, prod, clip, argmax, argmin,
    equal, not_equal, less, less_equal, greater, greater_equal,
    where, maximum, minimum,
)

# Lift all Ops static methods to package-level functions
add = Ops.add
subtract = Ops.subtract
multiply = Ops.multiply
divide = Ops.divide

__all__ = [
    "Tensor", "Variable", "Graph", "GradcheckError",
    "backward", "grad", "gradcheck", "hessian", "jacobian", "Ops", "DataType",
    "creation", "graph", "ops", "linalg", "math", "optim",
    "float64", "float32", "int64", "int32", "int16", "int8", "uint8",
    "add", "subtract", "multiply", "divide", "pow",
    "zeros", "ones", "full", "eye", "arange", "linspace",
    "dot", "matmul", "norm", "outer", "transpose", "reshape", "stack", "concat",
    "sum", "mean", "min", "max", "prod", "abs", "clip",
    "argmax", "argmin",
    "sqrt", "exp", "log", "sin", "cos", "tan",
    "arcsin", "arccos", "arctan",
    "sinh", "cosh", "arcsinh", "arccosh", "arctanh", "sign",
    "relu", "sigmoid", "tanh", "softplus", "softmax",
    "logsumexp", "log_softmax", "cross_entropy", "binary_cross_entropy",
    "std", "variance",
    "equal", "not_equal", "less", "less_equal", "greater", "greater_equal",
    "where", "maximum", "minimum",
]

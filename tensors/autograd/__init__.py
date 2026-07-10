"""Automatic differentiation for tensors.

Building the computation graph is **implicit** — operators automatically
register with a thread-local graph as you write math.

Execution (forward/backward) is **explicit** — you must use a
:class:`Graph` context manager::

    from tensors.autograd import Variable, Graph, dot, mean

    x = Variable([[1.0, 2.0]])
    w = Variable([[0.5], [0.5]])
    b = Variable([[0.0]])

    y = dot(x, w) + b
    loss = mean(y)

    with Graph() as g:
        g.backward(loss)

    print(w.grad)   # Tensor with shape (2, 1)
"""

from .variable import Variable, dot, sum, mean
from .graph import Graph
from .node import Node
from .edge import Edge
from ..ops import Add, Sub, Mul, Div, Neg, Dot, Sum, Mean

__all__ = [
    "Variable",
    "Graph",
    "Node",
    "Edge",
    "dot",
    "sum",
    "mean",
    "Add", "Sub", "Mul", "Div", "Neg", "Dot", "Sum", "Mean",
]

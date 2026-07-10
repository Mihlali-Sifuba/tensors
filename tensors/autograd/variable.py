"""Variable — a differentiable wrapper around Tensor.

Building the graph is implicit::

    x = Variable([1.0, 2.0])
    w = Variable([0.5, 0.5])
    y = x * w + 1.0

Execution is explicit::

    from tensors.autograd import Graph

    with Graph() as g:
        g.backward(y)

    print(w.grad)   # Tensor([0.5, 0.5])
"""

from ..tensor import Tensor
from ..ops import Ops
from .graph import _get_graph


class Variable:
    """A differentiable variable backed by a :class:`~tensors.Tensor`.

    Args:
        data: Initial data (Tensor, list, or number).
        name: Optional label for debugging and graph inspection.
    """

    def __init__(self, data, name=None):
        if isinstance(data, Tensor):
            self.data = data
        elif isinstance(data, Variable):
            self.data = data.data
        else:
            self.data = Tensor(data)

        self.grad = None
        self.name = name or f"v{id(self) & 0xFFFF:04x}"

        # Register as leaf node in the implicit graph
        self.node = _get_graph().add_node(label="var", output_var=self)

    # -- properties mirroring Tensor -----------------------------------

    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def size(self):
        return self.data.size

    # -- representation ------------------------------------------------

    def __repr__(self):
        grad_str = f", grad={self.grad}" if self.grad is not None else ""
        return f"Variable({self.data}{grad_str})"

    # -- operators (build graph implicitly) ----------------------------

    def __add__(self, other):
        g = _get_graph()
        if isinstance(other, Variable):
            out = Variable(Ops.add(self.data, other.data))
            op = g.add_node(label="add", output_var=out)
            g.add_edge(self.node, op)
            g.add_edge(other.node, op)
            return out
        out = Variable(self.data + other)
        op = g.add_node(label="add", output_var=out)
        g.add_edge(self.node, op)
        return out

    def __radd__(self, other):
        g = _get_graph()
        out = Variable(self.data + other)
        op = g.add_node(label="add", output_var=out)
        g.add_edge(self.node, op)
        return out

    def __sub__(self, other):
        g = _get_graph()
        if isinstance(other, Variable):
            out = Variable(Ops.subtract(self.data, other.data))
            op = g.add_node(label="sub", output_var=out)
            g.add_edge(self.node, op)
            g.add_edge(other.node, op)
            return out
        out = Variable(self.data - other)
        op = g.add_node(label="sub", output_var=out)
        g.add_edge(self.node, op)
        return out

    def __rsub__(self, other):
        g = _get_graph()
        out = Variable(-self.data + other)
        op = g.add_node(label="neg", output_var=out)
        g.add_edge(self.node, op)
        return out

    def __mul__(self, other):
        g = _get_graph()
        if isinstance(other, Variable):
            out = Variable(Ops.multiply(self.data, other.data))
            op = g.add_node(label="mul", output_var=out)
            g.add_edge(self.node, op)
            g.add_edge(other.node, op)
            return out
        out = Variable(self.data * other)
        op = g.add_node(label="mul", output_var=out, scalar=other)
        g.add_edge(self.node, op)
        return out

    def __rmul__(self, other):
        g = _get_graph()
        out = Variable(self.data * other)
        op = g.add_node(label="mul", output_var=out, scalar=other)
        g.add_edge(self.node, op)
        return out

    def __truediv__(self, other):
        g = _get_graph()
        if isinstance(other, Variable):
            out = Variable(Ops.divide(self.data, other.data))
            op = g.add_node(label="div", output_var=out)
            g.add_edge(self.node, op)
            g.add_edge(other.node, op)
            return out
        out = Variable(self.data / other)
        op = g.add_node(label="div", output_var=out, scalar=other)
        g.add_edge(self.node, op)
        return out

    def __neg__(self):
        g = _get_graph()
        out = Variable(Ops.multiply(self.data, -1))
        op = g.add_node(label="neg", output_var=out)
        g.add_edge(self.node, op)
        return out

    def __getitem__(self, key):
        return Variable(self.data[key])


# -- free functions that return Variables (graph-aware) ---------------

def dot(a, b):
    """Matrix multiplication between two Variables."""
    b = b if isinstance(b, Variable) else Variable(b)
    g = _get_graph()
    out = Variable(Ops.dot(a.data, b.data))
    op = g.add_node(label="dot", output_var=out)
    g.add_edge(a.node, op)
    g.add_edge(b.node, op)
    return out


def sum(var):
    """Sum of all elements."""
    g = _get_graph()
    out = Variable(Ops.sum(var.data))
    op = g.add_node(label="sum", output_var=out)
    g.add_edge(var.node, op)
    return out


def mean(var):
    """Mean of all elements."""
    g = _get_graph()
    out = Variable(Ops.mean(var.data))
    op = g.add_node(label="mean", output_var=out)
    g.add_edge(var.node, op)
    return out

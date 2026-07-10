"""Variable — a differentiable wrapper around Tensor.

Eager operations retain differentiation history::

    x = Variable([1.0, 2.0])
    w = Variable([0.5, 0.5])
    y = x * w + 1.0

Differentiation starts from a chosen output::

    from tensors import backward

    backward(y)

    print(w.grad)   # Tensor([1.0, 2.0])
"""

from .tensor import Tensor
from .ops import Ops
from .ops import Add, Sub, Mul, Div, Neg, Slice
from .graph.state import get_graph_state


class Variable:
    """A differentiable variable backed by a :class:`~tensors.Tensor`.

    Args:
        data: Initial data (Tensor, list, or number).
        name: Optional label for debugging and graph inspection.
        requires_grad: Whether gradients should be accumulated for this leaf.
    """

    def __init__(self, data, name=None, requires_grad=True, _register=True):
        if isinstance(data, Tensor):
            self.data = data
        elif isinstance(data, Variable):
            self.data = data.data
        else:
            self.data = Tensor(data)

        self.grad = None
        self.name = name or f"v{id(self) & 0xFFFF:04x}"
        self.requires_grad = requires_grad

        if requires_grad and self.dtype.typecode not in {"f", "d"}:
            raise ValueError("Only floating-point Variables can require gradients")

        self.node = None
        if _register:
            self.node = get_graph_state().add_node(label="var", output_var=self)

    @classmethod
    def _from_operation(cls, data, label, op_cls, inputs, **kwargs):
        """Create a result Variable owned by its operation node."""
        graph = get_graph_state()
        out = cls(
            data,
            requires_grad=any(var.requires_grad for var in inputs),
            _register=False,
        )
        node = graph.add_node(label=label, output_var=out, op_cls=op_cls, **kwargs)
        out.node = node
        for index, var in enumerate(inputs):
            graph.add_edge(var.node, node, label=f"input_{index}")
        return out

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
        if isinstance(other, Variable):
            return self._from_operation(
                Ops.add(self.data, other.data), "add", Add, [self, other]
            )
        return self._from_operation(
            Ops.add(self.data, other), "add", Add, [self], scalar=other
        )

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        if isinstance(other, Variable):
            return self._from_operation(
                Ops.subtract(self.data, other.data), "sub", Sub, [self, other]
            )
        return self._from_operation(
            Ops.subtract(self.data, other), "sub", Sub, [self], scalar=other
        )

    def __rsub__(self, other):
        return (-self) + other

    def __mul__(self, other):
        if isinstance(other, Variable):
            return self._from_operation(
                Ops.multiply(self.data, other.data), "mul", Mul, [self, other]
            )
        return self._from_operation(
            Ops.multiply(self.data, other), "mul", Mul, [self], scalar=other
        )

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        if isinstance(other, Variable):
            return self._from_operation(
                Ops.divide(self.data, other.data), "div", Div, [self, other]
            )
        return self._from_operation(
            Ops.divide(self.data, other), "div", Div, [self], scalar=other
        )

    def __rtruediv__(self, other):
        return self._from_operation(
            Div.forward_reverse(self.data, other),
            "div",
            Div,
            [self],
            scalar=other,
            reverse=True,
        )

    def __neg__(self):
        return self._from_operation(Ops.neg(self.data), "neg", Neg, [self])

    def __getitem__(self, key):
        return self._from_operation(
            Slice.forward(self.data, key), "slice", Slice, [self], key=key
        )

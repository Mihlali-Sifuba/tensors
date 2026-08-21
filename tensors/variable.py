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

from .dtype import DataType
from .tensor import Tensor
from .ops import Ops
from .ops import Add, Sub, Mul, Div, Pow, Neg, Slice, Cast
from .graph.state import get_graph_state


class Variable:
    """A differentiable variable backed by a :class:`~tensors.Tensor`.

    Args:
        data: Initial data (Tensor, list, or number).
        name: Optional label for debugging and graph inspection.
        requires_grad: Whether gradients should be accumulated for this leaf.
    """

    def __init__(self, data, name=None, requires_grad=True, _register=True):
        self._data_generation = 0
        self.requires_grad = requires_grad
        self.data = data.data if isinstance(data, Variable) else data
        self.grad = None
        self.name = name or f"v{id(self) & 0xFFFF:04x}"

        self.node = None
        if _register:
            self.node = get_graph_state().add_node(label="var", output_var=self)

    @classmethod
    def _from_operation(
        cls,
        data,
        label,
        op_cls,
        inputs,
        *,
        _scalar_operand=False,
        **kwargs,
    ):
        """Create a result Variable owned by its operation node."""
        graph = get_graph_state()
        out = cls(
            data,
            requires_grad=any(var.requires_grad for var in inputs),
            _register=False,
        )
        node = graph.add_node(
            label=label,
            output_var=out,
            op_cls=op_cls,
            _scalar_operand=_scalar_operand,
            **kwargs,
        )
        out.node = node
        for index, var in enumerate(inputs):
            graph.add_edge(var.node, node, label=f"input_{index}")
        node.capture_states()
        return out

    # -- properties mirroring Tensor -----------------------------------

    @property
    def requires_grad(self):
        """Whether reverse-mode derivatives should flow into this variable."""
        return self._requires_grad

    @requires_grad.setter
    def requires_grad(self, value):
        if not isinstance(value, bool):
            raise TypeError("requires_grad must be a bool")
        if (
            value
            and hasattr(self, "_data")
            and self._data.dtype.typecode not in {"f", "d"}
        ):
            raise ValueError("Only floating-point Variables can require gradients")
        self._requires_grad = value

    @property
    def data(self):
        """Tensor currently holding this variable's eager value."""
        return self._data

    @data.setter
    def data(self, value):
        """Replace the eager value and invalidate computations using the old one."""
        tensor = value if isinstance(value, Tensor) else Tensor(value)
        if getattr(self, "requires_grad", False) and tensor.dtype.typecode not in {
            "f", "d",
        }:
            raise ValueError("Only floating-point Variables can require gradients")
        self._data = tensor
        self._data_generation += 1

    def _mutation_state(self):
        """Return the value state recorded by operation nodes."""
        return (
            self._data_generation,
            self.data.version,
            self.data.shape,
            self.data.ndim,
            self.data.dtype.typecode,
            self.requires_grad,
        )

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

    def __eq__(self, other: object) -> bool:
        """Variables compare by graph identity, not by their current values."""
        return self is other

    def __ne__(self, other: object) -> bool:
        """Return whether two references identify different Variables."""
        return not self.__eq__(other)

    __hash__ = object.__hash__

    # -- operators (build graph implicitly) ----------------------------

    def __add__(self, other):
        if isinstance(other, Tensor):
            other = Variable(other, requires_grad=False)
        if isinstance(other, Variable):
            return self._from_operation(
                Ops.add(self.data, other.data), "add", Add, [self, other]
            )
        return self._from_operation(
            Ops.add(self.data, other), "add", Add, [self],
            _scalar_operand=True, scalar=other
        )

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        if isinstance(other, Tensor):
            other = Variable(other, requires_grad=False)
        if isinstance(other, Variable):
            return self._from_operation(
                Ops.subtract(self.data, other.data), "sub", Sub, [self, other]
            )
        return self._from_operation(
            Ops.subtract(self.data, other), "sub", Sub, [self],
            _scalar_operand=True, scalar=other
        )

    def __rsub__(self, other):
        return (-self) + other

    def __mul__(self, other):
        if isinstance(other, Tensor):
            other = Variable(other, requires_grad=False)
        if isinstance(other, Variable):
            return self._from_operation(
                Ops.multiply(self.data, other.data), "mul", Mul, [self, other]
            )
        return self._from_operation(
            Ops.multiply(self.data, other), "mul", Mul, [self],
            _scalar_operand=True, scalar=other
        )

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        if isinstance(other, Tensor):
            other = Variable(other, requires_grad=False)
        if isinstance(other, Variable):
            return self._from_operation(
                Ops.divide(self.data, other.data), "div", Div, [self, other]
            )
        return self._from_operation(
            Ops.divide(self.data, other), "div", Div, [self],
            _scalar_operand=True, scalar=other
        )

    def __rtruediv__(self, other):
        if isinstance(other, Tensor):
            numerator = Variable(other, requires_grad=False)
            return numerator / self
        return self._from_operation(
            Div.forward_reverse(self.data, other),
            "div",
            Div,
            [self],
            _scalar_operand=True,
            scalar=other,
            reverse=True,
        )

    def __pow__(self, other):
        if isinstance(other, Variable):
            return self._from_operation(
                Pow.forward(self.data, other.data),
                "pow",
                Pow,
                [self, other],
                differentiate_base=self.requires_grad,
                differentiate_exponent=other.requires_grad,
            )
        if isinstance(other, Tensor):
            exponent = Variable(other, requires_grad=False)
            return self._from_operation(
                Pow.forward(self.data, exponent.data),
                "pow",
                Pow,
                [self, exponent],
                differentiate_base=self.requires_grad,
                differentiate_exponent=False,
            )
        return self._from_operation(
            Pow.forward(self.data, other), "pow", Pow, [self],
            _scalar_operand=True, scalar=other
        )

    def __rpow__(self, other):
        if isinstance(other, Tensor):
            base = Variable(other, requires_grad=False)
            return base ** self
        if not isinstance(other, (int, float)):
            return NotImplemented
        return self._from_operation(
            Pow.forward_reverse(self.data, other),
            "pow",
            Pow,
            [self],
            _scalar_operand=True,
            scalar=other,
            reverse=True,
        )

    def __neg__(self):
        return self._from_operation(Ops.neg(self.data), "neg", Neg, [self])

    def __matmul__(self, other):
        from .linalg import matmul
        return matmul(self, other)

    def __rmatmul__(self, other):
        from .linalg import matmul
        return matmul(other, self)

    def __getitem__(self, key):
        return self._from_operation(
            Slice.forward(self.data, key), "slice", Slice, [self], key=key
        )

    def astype(self, dtype: str | DataType) -> "Variable":
        """Return a differentiable copy converted to ``dtype``."""
        result = self.data.astype(dtype)
        return self._from_operation(
            result,
            "astype",
            Cast,
            [self],
            dtype=result.dtype,
        )

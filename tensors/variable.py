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

from __future__ import annotations

from collections.abc import Iterable
from types import NotImplementedType
from typing import Any

from ._typing import TensorData, TensorIndex, TensorLike, TensorOperand, VariableData
from .dtype import DataType, result_dtype
from .shape import Shape
from .tensor import Tensor
from .ops import Add, Sub, Mul, Div, Pow, Neg, Slice, Cast
from .ops.pow import _power_dtype
from .graph.state import get_graph_state


_SCALAR_SHAPE = Shape()
_INPUT_LABELS = ("input_0", "input_1", "input_2", "input_3", "input_4")


def _operand_labels(count: int) -> tuple[str, ...]:
    """Return the edge labels naming an operation's ordered operands."""
    if count <= len(_INPUT_LABELS):
        return _INPUT_LABELS[:count]
    return tuple(f"input_{index}" for index in range(count))


class Variable:
    """A differentiable variable backed by a :class:`~tensors.Tensor`.

    Args:
        data: Initial data (Tensor, list, or number).
        name: Optional label for debugging and graph inspection.
        requires_grad: Whether gradients should be accumulated for this leaf.
    """

    def __init__(
        self,
        data: VariableData,
        name: str | None = None,
        requires_grad: bool = True,
    ) -> None:
        self._data_generation = 0
        self.requires_grad = requires_grad
        self.data = data.data if isinstance(data, Variable) else data
        self.grad = None
        self.name = name or f"v{id(self) & 0xFFFF:04x}"
        # Execution state, not graph structure: the forward pass states used
        # to reject differentiating a mutated computation, and the reusable
        # Computation compiled from the graph rooted at this Variable.
        self._forward_state: Any = None
        self._cached_computation: Any = None

        self.node = get_graph_state().add_variable_node(self)

    @classmethod
    def _record_operation(cls, data, operation, inputs):
        """Record an executed ``operation`` and return its result Variable.

        The operation has already run: this writes it into graph history by
        creating the result Variable, adding the operation vertex, joining
        each operand to it, joining it to the result, and capturing the
        result's forward state.

        The recorded topology is always
        ``VariableNode -> OperationNode -> VariableNode``: every operand
        arrives through an incoming edge and the result leaves through the
        single outgoing edge.
        """
        graph = get_graph_state()
        result = cls(
            data,
            requires_grad=any(operand.requires_grad for operand in inputs),
        )
        node = graph.add_operation_node(operation)
        for label, operand in zip(_operand_labels(len(inputs)), inputs):
            graph.add_edge(operand.node, node, label=label)
        graph.add_edge(node, result.node, label="result")
        result._capture_forward_state(inputs)
        return result

    def _capture_forward_state(self, operands: Iterable[Variable]) -> None:
        """Remember the operand and result states of the forward pass.

        The state describes values rather than topology, so it takes the
        operand Variables directly and execution never has to consult the
        graph to refresh it.
        """
        if not self.requires_grad:
            self._forward_state = None
            return
        self._forward_state = (
            tuple(operand._mutation_state() for operand in operands),
            self._mutation_state(),
        )

    # -- properties mirroring Tensor -----------------------------------

    @property
    def requires_grad(self) -> bool:
        """Whether reverse-mode derivatives should flow into this variable."""
        return self._requires_grad

    @requires_grad.setter
    def requires_grad(self, value: bool) -> None:
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
    def data(self) -> Tensor:
        """Tensor currently holding this variable's eager value."""
        return self._data

    @data.setter
    def data(self, value: TensorData) -> None:
        """Replace the eager value and invalidate computations using the old one."""
        tensor = value if isinstance(value, Tensor) else Tensor(value)
        if getattr(self, "requires_grad", False) and tensor.dtype.typecode not in {
            "f", "d",
        }:
            raise ValueError("Only floating-point Variables can require gradients")
        self._data = tensor
        self._data_generation += 1

    def _replace_data_from_replay(self, tensor: Tensor) -> None:
        """Adopt a validated internal replay result without public coercion."""
        self._data = tensor
        self._data_generation += 1

    def _mutation_state(self):
        """Return the value state recorded by operation nodes."""
        # Read through the attributes directly: every recorded operand is
        # re-checked on each replay and differentiation pass.
        return (
            self._data_generation,
            self._data._version,
            self._requires_grad,
        )

    @property
    def shape(self) -> Shape:
        return self.data.shape

    @property
    def ndim(self) -> int:
        return self.data.ndim

    @property
    def dtype(self) -> DataType:
        return self.data.dtype

    @property
    def size(self) -> int:
        return self.data.size

    # -- representation ------------------------------------------------

    def __repr__(self) -> str:
        grad_str = f", grad={self.grad}" if self.grad is not None else ""
        return f"Variable({self.data}{grad_str})"

    def __eq__(self, other: object) -> bool:
        """Variables compare by graph identity, not by their current values."""
        return self is other

    def __ne__(self, other: object) -> bool:
        """Return whether two references identify different Variables."""
        return not self.__eq__(other)

    __hash__ = object.__hash__

    def __bool__(self) -> bool:
        """Prevent Variables from silently behaving like truthy objects."""
        raise TypeError(
            "Cannot convert a Variable to a Python bool. "
            "Use variable.data.item() for scalar Variables or "
            "variable.size != 0 for emptiness checks."
        )

    # -- operators (build graph implicitly) ----------------------------
    #
    # Each operator resolves its own operand and records its own invocation.
    # A Python scalar becomes a non-gradient scalar Variable so the value
    # enters the graph structurally instead of hiding inside the operation,
    # typed by the dtype the scalar promotion rules select so that the
    # resulting tensor-tensor promotion reproduces it exactly.

    def __add__(self, other: TensorOperand) -> Variable:
        dtype = result_dtype(self.dtype, other)
        if isinstance(other, Variable):
            operand = other
        elif isinstance(other, Tensor):
            operand = Variable(other, requires_grad=False)
        else:
            operand = Variable(
                Tensor._from_values((other,), dtype, _SCALAR_SHAPE),
                requires_grad=False,
            )

        operation = Add()
        return self._record_operation(
            operation.forward(self.data, operand.data),
            operation,
            (self, operand),
        )

    def __radd__(self, other: int | float | Tensor) -> Variable:
        return self + other

    def __sub__(self, other: TensorOperand) -> Variable:
        dtype = result_dtype(self.dtype, other)
        if isinstance(other, Variable):
            operand = other
        elif isinstance(other, Tensor):
            operand = Variable(other, requires_grad=False)
        else:
            operand = Variable(
                Tensor._from_values((other,), dtype, _SCALAR_SHAPE),
                requires_grad=False,
            )

        operation = Sub()
        return self._record_operation(
            operation.forward(self.data, operand.data),
            operation,
            (self, operand),
        )

    def __rsub__(self, other: int | float | Tensor) -> Variable:
        return (-self) + other

    def __mul__(self, other: TensorOperand) -> Variable:
        dtype = result_dtype(self.dtype, other)
        if isinstance(other, Variable):
            operand = other
        elif isinstance(other, Tensor):
            operand = Variable(other, requires_grad=False)
        else:
            operand = Variable(
                Tensor._from_values((other,), dtype, _SCALAR_SHAPE),
                requires_grad=False,
            )

        operation = Mul()
        return self._record_operation(
            operation.forward(self.data, operand.data),
            operation,
            (self, operand),
        )

    def __rmul__(self, other: int | float | Tensor) -> Variable:
        return self * other

    def __truediv__(self, other: TensorOperand) -> Variable:
        dtype = result_dtype(self.dtype, other, division=True)
        if isinstance(other, Variable):
            operand = other
        elif isinstance(other, Tensor):
            operand = Variable(other, requires_grad=False)
        else:
            operand = Variable(
                Tensor._from_values((other,), dtype, _SCALAR_SHAPE),
                requires_grad=False,
            )

        operation = Div()
        return self._record_operation(
            operation.forward(self.data, operand.data),
            operation,
            (self, operand),
        )

    def __rtruediv__(self, other: int | float | Tensor) -> Variable:
        # Operand order carries the semantics: the numerator is input_0.
        dtype = result_dtype(self.dtype, other, division=True)
        if isinstance(other, Variable):
            numerator = other
        elif isinstance(other, Tensor):
            numerator = Variable(other, requires_grad=False)
        else:
            numerator = Variable(
                Tensor._from_values((other,), dtype, _SCALAR_SHAPE),
                requires_grad=False,
            )

        operation = Div()
        return self._record_operation(
            operation.forward(numerator.data, self.data),
            operation,
            (numerator, self),
        )

    def __pow__(self, other: TensorOperand) -> Variable:
        dtype = _power_dtype(self.data, other)
        if isinstance(other, Variable):
            exponent = other
        elif isinstance(other, Tensor):
            exponent = Variable(other, requires_grad=False)
        else:
            exponent = Variable(
                Tensor._from_values((other,), dtype, _SCALAR_SHAPE),
                requires_grad=False,
            )

        operation = Pow()
        return self._record_operation(
            operation.forward(self.data, exponent.data),
            operation,
            (self, exponent),
        )

    def __rpow__(
        self,
        other: int | float | Tensor,
    ) -> Variable | NotImplementedType:
        if not isinstance(other, (int, float, Tensor)) or isinstance(other, bool):
            return NotImplemented
        # Operand order carries the semantics: the base is input_0.
        dtype = result_dtype(self.dtype, other)
        if isinstance(other, Tensor):
            base = Variable(other, requires_grad=False)
        else:
            base = Variable(
                Tensor._from_values((other,), dtype, _SCALAR_SHAPE),
                requires_grad=False,
            )

        operation = Pow()
        return self._record_operation(
            operation.forward(base.data, self.data),
            operation,
            (base, self),
        )

    def __neg__(self) -> Variable:
        operation = Neg()
        return self._record_operation(
            operation.forward(self.data),
            operation,
            (self,),
        )

    def __abs__(self) -> Variable:
        from .math import abs
        return abs(self)

    def __matmul__(self, other: TensorLike) -> Variable:
        from .linalg import matmul
        return matmul(self, other)

    def __rmatmul__(self, other: TensorLike) -> Variable:
        from .linalg import matmul
        return matmul(other, self)

    def __getitem__(self, key: TensorIndex) -> Variable:
        operation = Slice(key=key)
        return self._record_operation(
            operation.forward(self.data),
            operation,
            (self,),
        )

    def astype(self, dtype: str | DataType) -> "Variable":
        """Return a differentiable copy converted to ``dtype``."""
        result = self.data.astype(dtype)
        return self._record_operation(result, Cast(dtype=result.dtype), (self,))

"""Applying an operation to the values an expression is written over.

An operation reaches the graph in one of two ways, and the operands decide
which. An expression over runtime :class:`~tensors.Variable` values is a
calculation: the operation is recorded and immediately compiled and executed.
An expression over :class:`~tensors.graph.node.VariableNode` values is a
description of a graph: the operation is recorded and nothing runs, because
the values those vertices name may not exist yet.

Both record the same topology through
:meth:`~tensors.graph.state.GraphState.record_operation`, which stays the one
owner of what an operation looks like in the graph. This module only decides
which of the two applications a caller asked for.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, NoReturn, TypeGuard

from .node import VariableNode
from .state import get_graph_state

if TYPE_CHECKING:
    from .._typing import GraphOperand, TensorLike
    from ..ops.operation import Operation
    from ..tensor import Tensor
    from ..variable import Variable


def is_graph_operand(value: object) -> TypeGuard[GraphOperand]:
    """Whether ``value`` names a graph value rather than a plain Tensor."""
    from ..variable import Variable

    return isinstance(value, (Variable, VariableNode))


def as_graph_operand(value: TensorLike | VariableNode) -> GraphOperand:
    """Return ``value`` as an operand an expression can be written over.

    A graph value takes part as itself. Any other value already exists, so
    it enters the graph as a non-gradient leaf — except a bare Python
    scalar, which is rejected so that no constant enters the graph with a
    guessed dtype.
    """
    from ..variable import Variable

    if isinstance(value, (Variable, VariableNode)):
        return value
    if isinstance(value, (int, float)):
        _reject_guessed_constant(value)
    return Variable(value, requires_grad=False)


def structural_node(operand: GraphOperand | Tensor) -> VariableNode:
    """Return the vertex ``operand`` takes part in a structural expression as.

    A runtime Variable is read as the vertex it was materialized against, so
    a structural expression never touches the value it holds, and a Tensor
    becomes a non-gradient leaf because it is a value that already exists.

    A Python scalar is rejected rather than guessed at: an eager scalar is
    typed by promotion against the value beside it, and a structural
    expression has not calculated that value.
    """
    from ..tensor import Tensor
    from ..variable import Variable

    if isinstance(operand, VariableNode):
        return operand
    if isinstance(operand, Variable):
        return operand.node
    if isinstance(operand, Tensor):
        return Variable(operand, requires_grad=False).node
    _reject_guessed_constant(operand)


def _reject_guessed_constant(value: object) -> NoReturn:
    """Reject a value the graph could only record with a guessed dtype."""
    raise TypeError(
        "An operation operand must be a VariableNode, Variable or Tensor, "
        f"got {type(value).__name__}. A Python scalar would have to be "
        "recorded as a constant whose dtype depends on the value beside it, "
        "which the graph has not calculated."
    )


def record_structurally(
    operation: Operation,
    operands: Sequence[GraphOperand | Tensor],
) -> VariableNode:
    """Record ``operation`` over ``operands`` and return its result vertex.

    Nothing is calculated: the operands are read as graph identities, the
    topology is recorded, and the result vertex comes back unbound.
    """
    return get_graph_state().record_operation(
        operation,
        tuple(structural_node(operand) for operand in operands),
    )


def apply_operation(
    operation: Operation,
    operands: Sequence[GraphOperand],
) -> Variable | VariableNode:
    """Apply ``operation`` as the kinds of its operands imply.

    Every operand being a runtime value makes this a calculation, so the
    recorded fragment is compiled, executed, and materialized. A single
    structural operand makes the whole expression structural, because a value
    that does not exist yet cannot take part in one that runs now.
    """
    from ..variable import Variable

    runtime = [
        operand for operand in operands if isinstance(operand, Variable)
    ]
    if len(runtime) != len(operands):
        return record_structurally(operation, operands)
    return Variable._apply_operation(operation, runtime)


__all__ = [
    "apply_operation",
    "as_graph_operand",
    "is_graph_operand",
    "structural_node",
    "record_structurally",
]

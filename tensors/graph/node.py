"""Vertices in a computational graph.

A graph alternates between the two concrete vertex types::

    VariableNode -> OperationNode -> VariableNode

:class:`Node` holds only what every vertex shares: an identity and its
connectivity. Variable-specific and operation-specific state belongs to the
concrete subclasses, and execution state belongs to
:class:`~tensors.graph.computation.Computation`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from weakref import ReferenceType, ref

if TYPE_CHECKING:
    from ..variable import Variable
    from .edge import Edge
    from ..ops.operation import Operation


class Node:
    """Identity and connectivity shared by every computational graph vertex."""

    _next_id = 0

    __slots__ = ("id", "_in_edges", "_out_edge_references", "__weakref__")

    def __init__(self) -> None:
        self.id = Node._next_id
        Node._next_id += 1
        self._in_edges: list[Edge] = []
        # Incoming edges are owned strongly because a result must retain every
        # dependency it needs for replay and differentiation. Outgoing edges
        # are weak so a persistent leaf (for example, a model parameter) does
        # not retain every result ever calculated from it.
        self._out_edge_references: list[ReferenceType[Edge]] = []

    def _add_out_edge(self, edge: Edge) -> None:
        """Register an outgoing edge without owning its target computation."""
        self._out_edge_references.append(ref(edge))

    def _replace_out_edges(self, edges: list[Edge] | tuple[Edge, ...]) -> None:
        """Restore the live outgoing edges used by an isolated trace."""
        self._out_edge_references = [ref(edge) for edge in edges]

    @property
    def _out_edges(self) -> list[Edge]:
        """Return live outgoing edges while pruning collected references."""
        live = []
        references = []
        for reference in self._out_edge_references:
            edge = reference()
            if edge is not None:
                live.append(edge)
                references.append(reference)
        if len(references) != len(self._out_edge_references):
            self._out_edge_references = references
        return live

    @property
    def label(self) -> str:
        """Return a short description used for graph inspection."""
        return "node"

    @property
    def inputs(self) -> list[Node]:
        """Return predecessor nodes."""
        return [edge.source for edge in self._in_edges]

    @property
    def outputs(self) -> list[Node]:
        """Return successor nodes."""
        return [edge.target for edge in self._out_edges]

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.label}, #{self.id})"

    def __hash__(self) -> int:
        return self.id

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Node) and self.id == other.id


class VariableNode(Node):
    """The graph representation of exactly one :class:`~tensors.Variable`.

    ``variable.node.variable is variable`` holds for every Variable, including
    leaves, normalized Tensor and scalar operands, and operation results. The
    reference is strong in both directions so a retained result keeps the
    upstream Variables that its replay and differentiation require. The
    resulting cycle is ordinary garbage, so an unreachable computation is still
    collectable.
    """

    __slots__ = ("variable",)

    def __init__(self, variable: Variable) -> None:
        super().__init__()
        self.variable = variable

    @property
    def label(self) -> str:
        """Return the inspection label shared by every Variable vertex."""
        return "var"

    @property
    def producer(self) -> OperationNode | None:
        """Return the operation vertex that calculated this Variable, if any."""
        edges = self._in_edges
        return edges[0].source if edges else None  # type: ignore[return-value]


class OperationNode(Node):
    """The graph representation of one concrete :class:`Operation` invocation.

    Operands arrive through incoming edges and the result leaves through a
    single outgoing edge. The node never stores those Variables directly, and
    it never interprets the operation's configuration.
    """

    __slots__ = ("operation",)

    def __init__(self, operation: Operation) -> None:
        super().__init__()
        self.operation = operation

    @property
    def label(self) -> str:
        """Return the recorded operation's short name."""
        return self.operation.name

    @property
    def operands(self) -> tuple[Variable, ...]:
        """Return the operand Variables named by this node's incoming edges."""
        return tuple(edge.source.variable for edge in self._in_edges)

    @property
    def result(self) -> Any:
        """Return the Variable named by this node's single outgoing edge.

        Outgoing edges are weak, so this returns ``None`` once the result has
        been collected. :class:`Computation` resolves the relationship once at
        construction and holds the Variable it needs from then on.
        """
        edges = self._out_edges
        return edges[0].target.variable if edges else None


__all__ = ["Node", "OperationNode", "VariableNode"]

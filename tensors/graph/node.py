"""Vertices in a computational graph.

A graph alternates between the two concrete vertex types::

    VariableNode -> OperationNode -> VariableNode

:class:`Node` holds only what every vertex shares: an identity and its
connectivity. Variable-specific and operation-specific state belongs to the
concrete subclasses, and execution state belongs to
:class:`~tensors.graph.computation.Computation`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from weakref import ReferenceType, ref

if TYPE_CHECKING:
    from .._typing import VariableData
    from ..variable import Variable
    from .edge import Edge
    from ..ops.operation import Operation


class UnboundVariableNodeError(RuntimeError):
    """Raised when a Variable vertex is read as a value it does not yet hold.

    A vertex names a value from the moment the graph names it, which can be
    long before a Computation calculates that value. Reading the value of an
    unmaterialized vertex is a lifecycle error, not a missing value, so it
    raises instead of reporting ``None``.
    """


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

    # A vertex is its own identity: ids are unique, so comparing them would
    # say exactly what comparing objects already says. Leaving the default
    # identity hash in place also keeps the dictionaries and sets that
    # compilation and execution key by vertex on the interpreter's fast path.



class VariableNode(Node):
    """The graph-level identity of one value in a computation.

    A vertex names a value; it does not require that value to exist yet. A
    leaf is recorded from a value that already exists, so it binds its
    :class:`~tensors.Variable` at construction. A value the graph only
    describes is created unbound: the vertex takes its place in the topology,
    compilation and execution are expressed over that structure, and the
    runtime Variable is materialized once execution produces its Tensor::

        VariableNode(a) ──input_0──┐
                                   ▼
                            OperationNode(Add())
                                   ▲
        VariableNode(b) ──input_1──┘
                                   │
                                result
                                   ▼
                            VariableNode(c)   # bound after Add has run

    Binding is one-time and symmetric: a vertex names at most one Variable
    and that Variable names it back, so ``variable.node.variable is variable``
    holds from materialization onwards for leaves, for normalized Tensor and
    scalar operands, and for operation results. The reference is strong in
    both directions so a retained result keeps the upstream Variables that its
    replay and differentiation require. The resulting cycle is ordinary
    garbage, so an unreachable computation is still collectable.
    """

    __slots__ = ("_variable",)

    def __init__(self, variable: Variable | None = None) -> None:
        super().__init__()
        self._variable: Variable | None = None
        if variable is not None:
            self.bind(variable)

    @property
    def label(self) -> str:
        """Return the inspection label shared by every Variable vertex."""
        return "var"

    @property
    def is_bound(self) -> bool:
        """Whether a runtime Variable has been materialized for this vertex."""
        return self._variable is not None

    @property
    def variable(self) -> Variable:
        """Return the runtime Variable this vertex was materialized as.

        Raises :class:`UnboundVariableNodeError` while the vertex is still
        only a graph identity: code that runs before materialization asks
        :attr:`is_bound` first rather than reading a value that does not
        exist.
        """
        variable = self._variable
        if variable is None:
            raise UnboundVariableNodeError(
                f"{self!r} names a value that has not been materialized"
            )
        return variable

    def bind(self, variable: Variable) -> None:
        """Materialize this vertex as ``variable``.

        Binding establishes both directions of the relationship at once, so a
        vertex and its Variable can never disagree about which value they
        name. Rebinding is rejected rather than silently replacing the first
        Variable: the identity a graph was built against stays the identity it
        executes with.
        """
        bound = self._variable
        if bound is not None:
            raise RuntimeError(
                f"{self!r} is already bound to Variable {bound.name!r}"
            )
        current = getattr(variable, "node", None)
        if current is not None and current is not self:
            raise RuntimeError(
                f"Variable {variable.name!r} is already bound to {current!r}"
            )
        self._variable = variable
        variable.node = self

    def materialize(
        self,
        data: VariableData,
        name: str | None = None,
        requires_grad: bool = True,
    ) -> Variable:
        """Create and bind the runtime Variable this vertex names.

        This closes the lifecycle: the graph named the value, execution
        calculated it, and the Variable carrying it is created against the
        vertex that was already there.
        """
        from ..variable import Variable

        return Variable(data, name, requires_grad, node=self)

    @property
    def producer(self) -> OperationNode | None:
        """Return the operation vertex that calculated this value, if any."""
        edges = self._in_edges
        return _as_operation_node(edges[0].source) if edges else None


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
    def operand_nodes(self) -> tuple[VariableNode, ...]:
        """Return the operand vertices named by this node's incoming edges.

        These are structure, so they are readable whether or not the operands
        have been materialized.
        """
        return tuple(_as_variable_node(edge.source) for edge in self._in_edges)

    @property
    def operands(self) -> tuple[Variable, ...]:
        """Return the operand Variables named by this node's incoming edges.

        Every operand must already be materialized; read
        :attr:`operand_nodes` instead to work with the structure alone.
        """
        return tuple(node.variable for node in self.operand_nodes)

    @property
    def result_node(self) -> VariableNode | None:
        """Return the vertex this operation produces into, if still live.

        Outgoing edges are weak, so this returns ``None`` once the result
        vertex has been collected.
        """
        edges = self._out_edges
        return _as_variable_node(edges[0].target) if edges else None

    @property
    def result(self) -> Variable | None:
        """Return the Variable named by this node's single outgoing edge.

        ``None`` once the result vertex has been collected; a live vertex that
        has not been materialized raises instead, so "no result any more" and
        "no value yet" stay distinguishable. :class:`Computation` resolves the
        relationship once at construction and holds the Variable it needs from
        then on.
        """
        node = self.result_node
        return node.variable if node is not None else None


def _as_variable_node(node: Node) -> VariableNode:
    """Return ``node`` as the Variable vertex the alternation requires."""
    if not isinstance(node, VariableNode):
        raise TypeError(f"Expected a VariableNode, got {type(node).__name__}")
    return node


def _as_operation_node(node: Node) -> OperationNode:
    """Return ``node`` as the operation vertex the alternation requires."""
    if not isinstance(node, OperationNode):
        raise TypeError(
            f"Expected an OperationNode, got {type(node).__name__}"
        )
    return node


__all__ = [
    "Node",
    "OperationNode",
    "UnboundVariableNodeError",
    "VariableNode",
]

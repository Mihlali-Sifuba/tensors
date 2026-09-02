"""A vertex in a computational graph."""

from __future__ import annotations

from functools import cache
from typing import TYPE_CHECKING, Any
from weakref import ReferenceType, ref

from .protocols import Operation

if TYPE_CHECKING:
    from .edge import Edge


@cache
def operation_methods(op_cls: type[Operation]) -> tuple[Any, Any, Any, Any]:
    """Validate and resolve an operation protocol once per operation class."""
    forward = getattr(op_cls, "forward", None)
    backward = getattr(op_cls, "backward", None)
    if not callable(forward) or not callable(backward):
        raise TypeError(
            "op_cls must provide callable forward() and backward() methods"
        )
    return (
        forward,
        getattr(op_cls, "forward_reverse", None),
        backward,
        getattr(op_cls, "backward_graph", None),
    )


class Node:
    """A leaf value or operation in a computational graph."""

    _next_id = 0

    __slots__ = (
        "id",
        "label",
        "output_var",
        "op_cls",
        "_scalar_operand",
        "args",
        "_in_edges",
        "_out_edge_references",
        "_input_states",
        "_output_state",
        "__weakref__",
    )

    def __init__(
        self,
        label: str | None = None,
        output_var: Any = None,
        op_cls: type[Operation] | None = None,
        _scalar_operand: bool = False,
        **kwargs: Any,
    ) -> None:
        self.id = Node._next_id
        Node._next_id += 1

        if op_cls is not None:
            operation_methods(op_cls)

        self.label = label
        self.output_var = output_var
        self.op_cls = op_cls
        self._scalar_operand = _scalar_operand
        self.args: dict[str, Any] = kwargs
        self._in_edges: list[Edge] = []
        # Incoming edges are owned strongly because an output must retain all
        # of its dependencies. Outgoing edges are weak so a persistent leaf
        # (for example, a model parameter) does not retain every old result.
        self._out_edge_references: list[ReferenceType[Edge]] = []
        self._input_states: tuple[Any, ...] = ()
        self._output_state: Any = None

    def capture_states(self) -> None:
        """Remember the eager input and output states of this operation."""
        if self.output_var is None or not self.output_var.requires_grad:
            self._input_states = ()
            self._output_state = None
            return
        self._input_states = tuple(
            (
                edge.source.output_var._mutation_state()
                if edge.source.output_var is not None
                else None
            )
            for edge in self._in_edges
        )
        self._output_state = (
            self.output_var._mutation_state()
            if self.output_var is not None
            else None
        )

    def changed_input(self) -> tuple[int, Any] | None:
        """Return the first input whose value changed since capture, if any."""
        if len(self._input_states) != len(self._in_edges):
            return 0, None
        for index, (edge, expected) in enumerate(
            zip(self._in_edges, self._input_states)
        ):
            variable = edge.source.output_var
            current = variable._mutation_state() if variable is not None else None
            if current != expected:
                return index, variable
        return None

    def output_changed(self) -> bool:
        """Return whether this operation's eager output was modified."""
        current = (
            self.output_var._mutation_state()
            if self.output_var is not None
            else None
        )
        return current != self._output_state

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
    def inputs(self) -> list[Node]:
        """Return predecessor nodes."""
        return [edge.source for edge in self._in_edges]

    @property
    def outputs(self) -> list[Node]:
        """Return successor nodes."""
        return [edge.target for edge in self._out_edges]

    def __repr__(self) -> str:
        return f"Node({self.label or '?'}, #{self.id})"

    def __hash__(self) -> int:
        return self.id

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Node) and self.id == other.id


__all__ = ["Node", "operation_methods"]

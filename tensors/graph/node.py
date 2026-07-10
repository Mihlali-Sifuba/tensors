"""A vertex in a computational graph."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .edge import Edge


class Node:
    """A leaf value or operation in a computational graph."""

    _next_id = 0

    def __init__(
        self,
        label: str | None = None,
        output_var: Any = None,
        op_cls: Any = None,
        **kwargs: Any,
    ) -> None:
        self.id = Node._next_id
        Node._next_id += 1

        self.label = label
        self.output_var = output_var
        self.op_cls = op_cls
        self.args: dict[str, Any] = kwargs
        self._in_edges: list[Edge] = []
        self._out_edges: list[Edge] = []

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


__all__ = ["Node"]

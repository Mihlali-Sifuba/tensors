"""A directed connection between computational graph nodes."""

from __future__ import annotations

from .node import Node


class Edge:
    """A data-flow connection from a source node to a target node."""

    def __init__(
        self,
        source: Node,
        target: Node,
        label: str | None = None,
    ) -> None:
        self.source = source
        self.target = target
        self.label = label

        source._out_edges.append(self)
        target._in_edges.append(self)

    def __repr__(self) -> str:
        label = f" {self.label!r}" if self.label else ""
        source = self.source.label or "?"
        target = self.target.label or "?"
        return f"Edge({source} → {target}{label})"


__all__ = ["Edge"]

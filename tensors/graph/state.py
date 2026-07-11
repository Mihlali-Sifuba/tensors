"""Thread-local state used while eager operations record graph history."""

from __future__ import annotations

import threading
from typing import Any

from .edge import Edge
from .node import Node


class GraphState:
    """Mutable collection of nodes and edges recorded by eager operations."""

    __slots__ = ("nodes", "edges")

    def __init__(self) -> None:
        self.nodes: list[Node] = []
        self.edges: list[Edge] = []

    def add_node(
        self,
        label: str,
        output_var: Any = None,
        op_cls: Any = None,
        **kwargs: Any,
    ) -> Node:
        node = Node(label=label, output_var=output_var, op_cls=op_cls, **kwargs)
        self.nodes.append(node)
        return node

    def add_edge(
        self,
        source: Node,
        target: Node,
        label: str | None = None,
    ) -> Edge:
        if source not in self.nodes:
            self.nodes.append(source)
        if target not in self.nodes:
            self.nodes.append(target)
        edge = Edge(source, target, label=label)
        self.edges.append(edge)
        return edge


_local = threading.local()


class TraceScope:
    """Internal tracing lifetime guard for nested Graph calls."""

    def __init__(self) -> None:
        depth = getattr(_local, "trace_depth", 0)
        self.outermost = depth == 0
        if self.outermost:
            reset_graph_state()
        _local.trace_depth = depth + 1
        self._closed = False

    def close(self) -> None:
        """Leave this trace scope."""
        if self._closed:
            return
        depth = getattr(_local, "trace_depth", 0)
        _local.trace_depth = max(depth - 1, 0)
        self._closed = True


def get_graph_state() -> GraphState:
    """Return the current thread's eager graph state."""
    if not hasattr(_local, "graph"):
        _local.graph = GraphState()
    return _local.graph


def reset_graph_state() -> None:
    """Replace the current thread's eager graph state."""
    _local.graph = GraphState()


__all__ = ["GraphState", "TraceScope", "get_graph_state", "reset_graph_state"]

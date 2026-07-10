"""Reusable computational graph functions and structural types."""

from typing import Any

from .node import Node
from .edge import Edge

__all__ = ["Graph", "Node", "Edge"]


def __getattr__(name: str) -> Any:
    """Load Graph lazily so autograd can import Node and Edge safely."""
    if name == "Graph":
        from .graph import Graph

        globals()[name] = Graph
        return Graph
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

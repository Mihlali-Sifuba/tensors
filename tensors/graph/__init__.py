"""Reusable computational graph functions and structural types."""

from typing import Any

from .node import Node
from .edge import Edge
from .computation import Computation, backward, grad

__all__ = ["Graph", "Computation", "Node", "Edge", "backward", "grad"]


def __getattr__(name: str) -> Any:
    """Load Graph lazily to avoid circular core imports."""
    if name == "Graph":
        from .graph import Graph

        globals()[name] = Graph
        return Graph
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

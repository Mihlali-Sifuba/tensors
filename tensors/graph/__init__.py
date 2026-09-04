"""Reusable computational graph functions and structural types."""

from typing import Any

from .node import Node, OperationNode, VariableNode
from .edge import Edge
from .computation import Computation
from .autograd import backward, grad
from .derivatives import hessian, jacobian
from .gradcheck import GradcheckError, gradcheck

__all__ = [
    "Graph", "Computation", "Node", "OperationNode", "VariableNode", "Edge",
    "GradcheckError", "backward", "grad", "gradcheck", "hessian", "jacobian",
]


def __getattr__(name: str) -> Any:
    """Load Graph lazily to avoid circular core imports."""
    if name == "Graph":
        from .graph import Graph

        globals()[name] = Graph
        return Graph
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

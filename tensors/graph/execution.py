"""Traversal and replay helpers for recorded computational graphs."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from ..tensor import Tensor
from .node import Node


def topological_sort(output_nodes: Iterable[Node]) -> list[Node]:
    """Return reachable nodes in dependency-first order."""
    order: list[Node] = []
    visited: set[Node] = set()

    def visit(node: Node) -> None:
        if node in visited:
            return
        visited.add(node)
        for edge in node._in_edges:
            visit(edge.source)
        order.append(node)

    for output_node in output_nodes:
        visit(output_node)
    return order


def replay(output_var: Any) -> Tensor:
    """Recompute a recorded output from its current leaf values."""
    values: dict[Any, Tensor] = {}
    for node in topological_sort([output_var.node]):
        if node.label == "var":
            values[node.output_var] = node.output_var.data
            continue
        result = _execute_node(node, values)
        node.output_var.data = result
        values[node.output_var] = result
    return values[output_var]


def _execute_node(node: Node, values: dict[Any, Tensor]) -> Tensor:
    inputs = [edge.source.output_var for edge in node._in_edges]
    args = [values[value] for value in inputs]

    if "scalar" in node.args:
        scalar = node.args["scalar"]
        if node.args.get("reverse", False):
            result = node.op_cls.forward_reverse(args[0], scalar)
        else:
            result = node.op_cls.forward(args[0], scalar)
    elif "key" in node.args:
        result = node.op_cls.forward(args[0], node.args["key"])
    else:
        result = node.op_cls.forward(*args)

    return result if isinstance(result, Tensor) else Tensor([result])


__all__ = ["replay", "topological_sort"]

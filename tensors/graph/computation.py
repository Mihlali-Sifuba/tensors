"""A recorded computation rooted at an output Variable."""

from __future__ import annotations

from array import array
from typing import TYPE_CHECKING, Any

from ..tensor import Tensor
from .node import Node

if TYPE_CHECKING:
    from ..variable import Variable


class Computation:
    """A concrete computation that owns its forward and backward passes."""

    def __init__(self, output: Any) -> None:
        if getattr(output, "node", None) is None:
            raise TypeError("Computation output must have a graph node")
        self.output = output

    @property
    def nodes(self) -> list[Node]:
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

        visit(self.output.node)
        return order

    def forward(self) -> Tensor:
        """Recompute the output from its current leaf values."""
        values: dict[Any, Tensor] = {}
        for node in self.nodes:
            if node.label == "var":
                values[node.output_var] = node.output_var.data
                continue

            result = self._execute_node(node, values)
            node.output_var.data = result
            values[node.output_var] = result
        return values[self.output]

    def backward(
        self,
        grad: Tensor | array | list[Any] | int | float | None = None,
    ) -> None:
        """Differentiate the output with respect to reachable Variables."""
        for node in self.nodes:
            if node.output_var is not None:
                node.output_var.grad = None

        if grad is None:
            typecode = (
                self.output.dtype.typecode
                if self.output.dtype.typecode in {"f", "d"}
                else "d"
            )
            seed_data = array(typecode, [1.0] * self.output.data.size)
            self.output.grad = Tensor(seed_data, shape=self.output.data.shape)
        else:
            seed = grad if isinstance(grad, Tensor) else Tensor(grad)
            if seed.shape != self.output.data.shape:
                raise ValueError(
                    f"Gradient shape {seed.shape} does not match output shape "
                    f"{self.output.data.shape}"
                )
            self.output.grad = seed

        for node in reversed(self.nodes):
            if node.label == "var" or node.op_cls is None:
                continue

            output = node.output_var
            if output.grad is None:
                continue

            input_data = [edge.source.output_var.data for edge in node._in_edges]
            gradients = node.op_cls.backward(output.grad, *input_data, **node.args)
            for edge, gradient in zip(node._in_edges, gradients):
                self._accumulate_gradient(edge.source.output_var, gradient)

    @staticmethod
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

    @staticmethod
    def _accumulate_gradient(variable: Variable, gradient: Tensor) -> None:
        if not variable.requires_grad:
            return
        if variable.grad is None:
            variable.grad = gradient
            return

        from ..ops import Add

        variable.grad = Add.forward(variable.grad, gradient)


def backward(
    output: Any,
    grad: Tensor | array | list[Any] | int | float | None = None,
) -> None:
    """Differentiate an output through its recorded Computation."""
    Computation(output).backward(grad)


__all__ = ["Computation", "backward"]

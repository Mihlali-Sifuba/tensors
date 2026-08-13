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
        *,
        create_graph: bool = False,
    ) -> None:
        """Differentiate the output with respect to reachable Variables."""
        if create_graph:
            seed = self._gradient_seed(grad, create_graph=True)
            gradients = self._backward_graph(seed)
            for node in self.nodes:
                variable = node.output_var
                if variable is not None:
                    variable.grad = gradients.get(variable)
            return

        for node in self.nodes:
            if node.output_var is not None:
                node.output_var.grad = None

        self.output.grad = self._gradient_seed(grad)

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

    def _gradient_seed(
        self,
        grad: Tensor | array | list[Any] | int | float | None,
        *,
        create_graph: bool = False,
    ) -> Any:
        """Return a validated upstream gradient, optionally as a Variable."""
        from ..variable import Variable

        if grad is None:
            typecode = (
                self.output.dtype.typecode
                if self.output.dtype.typecode in {"f", "d"}
                else "d"
            )
            seed = Tensor(
                [1.0] * self.output.data.size,
                dtype=typecode,
                shape=self.output.data.shape,
            )
        elif isinstance(grad, Variable):
            seed = grad
        else:
            seed = grad if isinstance(grad, Tensor) else Tensor(grad)

        seed_shape = seed.shape if isinstance(seed, Variable) else seed.shape
        if seed_shape != self.output.data.shape:
            raise ValueError(
                f"Gradient shape {seed_shape} does not match output shape "
                f"{self.output.data.shape}"
            )
        if create_graph:
            return seed if isinstance(seed, Variable) else Variable(seed, requires_grad=False)
        return seed.data if isinstance(seed, Variable) else seed

    def _backward_graph(self, seed: Any) -> dict[Any, Any]:
        """Build a differentiable reverse-mode gradient computation."""
        gradients: dict[Any, Any] = {self.output: seed}
        for node in reversed(self.nodes):
            if node.label == "var" or node.op_cls is None:
                continue

            output = node.output_var
            output_gradient = gradients.get(output)
            if output_gradient is None:
                continue
            backward_graph = getattr(node.op_cls, "backward_graph", None)
            inputs = [edge.source.output_var for edge in node._in_edges]
            if not any(input_variable.requires_grad for input_variable in inputs):
                continue
            if backward_graph is None:
                raise NotImplementedError(
                    f"Higher-order derivatives are not implemented for {node.label}"
                )
            input_gradients = backward_graph(output_gradient, *inputs, **node.args)
            for input_variable, input_gradient in zip(inputs, input_gradients):
                if not input_variable.requires_grad:
                    continue
                existing = gradients.get(input_variable)
                gradients[input_variable] = (
                    input_gradient if existing is None else existing + input_gradient
                )
        return gradients

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
        elif "axis" in node.args:
            axis = node.args.get("axis")
            keepdims = node.args.get("keepdims", False)
            result = node.op_cls.forward(*args, axis=axis, keepdims=keepdims)
        elif "shape" in node.args:
            result = node.op_cls.forward(args[0], node.args["shape"])
        elif "axes" in node.args:
            result = node.op_cls.forward(args[0], axes=node.args["axes"])
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
    *,
    create_graph: bool = False,
) -> None:
    """Differentiate an output through its recorded Computation."""
    Computation(output).backward(grad, create_graph=create_graph)


def grad(
    output: Any,
    inputs: Any,
    grad_outputs: Tensor | array | list[Any] | int | float | None = None,
    *,
    create_graph: bool = False,
) -> Any:
    """Return gradients of ``output`` with respect to one or more inputs.

    Set ``create_graph=True`` when the returned gradients will themselves be
    differentiated.
    """
    from ..variable import Variable

    single_input = isinstance(inputs, Variable)
    requested = (inputs,) if single_input else tuple(inputs)
    computation = Computation(output)
    if create_graph:
        seed = computation._gradient_seed(grad_outputs, create_graph=True)
        gradients = computation._backward_graph(seed)
        result = tuple(gradients.get(variable) for variable in requested)
    else:
        computation.backward(grad_outputs)
        result = tuple(variable.grad for variable in requested)
    return result[0] if single_input else result


__all__ = ["Computation", "backward", "grad"]

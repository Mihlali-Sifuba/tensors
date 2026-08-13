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
        stack = [(self.output.node, False)]
        while stack:
            node, expanded = stack.pop()
            if expanded:
                order.append(node)
                continue
            if node in visited:
                continue
            visited.add(node)
            stack.append((node, True))
            for edge in reversed(node._in_edges):
                if edge.source not in visited:
                    stack.append((edge.source, False))
        return order

    def forward(self) -> Tensor:
        """Recompute the output from its current leaf values."""
        values: dict[Any, Tensor] = {}
        for node in self.nodes:
            if node.op_cls is None:
                if node.output_var is None:
                    raise RuntimeError(
                        f"Leaf node {node!r} has no output Variable"
                    )
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
            if node.op_cls is None:
                continue

            output = node.output_var
            if output.grad is None:
                continue

            input_data = [edge.source.output_var.data for edge in node._in_edges]
            gradients = node.op_cls.backward(output.grad, *input_data, **node.args)
            gradients = self._validate_gradients(node, gradients, graph=False)
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
            if node.op_cls is None:
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
            input_gradients = self._validate_gradients(
                node, input_gradients, graph=True
            )
            for input_variable, input_gradient in zip(inputs, input_gradients):
                if not input_variable.requires_grad:
                    continue
                existing = gradients.get(input_variable)
                gradients[input_variable] = (
                    input_gradient if existing is None else existing + input_gradient
                )
        return gradients

    @staticmethod
    def _validate_gradients(node: Node, gradients: Any, *, graph: bool) -> tuple[Any, ...]:
        """Validate an operation's VJP result before propagating it."""
        from ..variable import Variable

        try:
            results = tuple(gradients)
        except TypeError as exc:
            raise TypeError(
                f"{node.label} backward must return one gradient per input"
            ) from exc

        expected = len(node._in_edges)
        if len(results) != expected:
            raise RuntimeError(
                f"{node.label} backward returned {len(results)} gradients for "
                f"{expected} inputs"
            )

        expected_type = Variable if graph else Tensor
        for index, (edge, gradient) in enumerate(zip(node._in_edges, results)):
            if not isinstance(gradient, expected_type):
                mode = "backward_graph" if graph else "backward"
                raise TypeError(
                    f"{node.label} {mode} gradient {index} must be a "
                    f"{expected_type.__name__}, got {type(gradient).__name__}"
                )
            expected_shape = edge.source.output_var.shape
            if gradient.shape != expected_shape:
                raise ValueError(
                    f"{node.label} backward gradient {index} has shape "
                    f"{gradient.shape}; expected {expected_shape}"
                )
        return results

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
        else:
            result = node.op_cls.forward(*args, **node.args)

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
    try:
        requested = (inputs,) if single_input else tuple(inputs)
    except TypeError as exc:
        raise TypeError(
            "grad inputs must be a Variable or an iterable of Variables"
        ) from exc
    if not requested:
        raise ValueError("grad requires at least one input Variable")
    for index, variable in enumerate(requested):
        if not isinstance(variable, Variable):
            raise TypeError(
                f"grad input {index} must be a Variable, got "
                f"{type(variable).__name__}"
            )
    computation = Computation(output)
    if create_graph:
        seed = computation._gradient_seed(grad_outputs, create_graph=True)
        gradients = computation._backward_graph(seed)
        result = tuple(gradients.get(variable) for variable in requested)
    else:
        for variable in requested:
            variable.grad = None
        computation.backward(grad_outputs)
        result = tuple(variable.grad for variable in requested)
    return result[0] if single_input else result


__all__ = ["Computation", "backward", "grad"]

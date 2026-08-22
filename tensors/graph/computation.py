"""A recorded computation rooted at an output Variable."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, overload

from .._typing import TensorLike
from ..tensor import Tensor
from .node import Node
from .protocols import HigherOrderOperation, ReverseOperation

if TYPE_CHECKING:
    from ..variable import Variable


class Computation:
    """A concrete computation that owns its forward and backward passes."""

    def __init__(self, output: Variable) -> None:
        if getattr(output, "node", None) is None:
            raise TypeError("Computation output must have a graph node")
        self.output = output
        self._nodes = self._dependency_order(output.node)
        self._released = False

    @staticmethod
    def _dependency_order(output_node: Node) -> tuple[Node, ...]:
        """Calculate and cache the dependency-first traversal for an output."""
        order: list[Node] = []
        visited: set[Node] = set()
        stack = [(output_node, False)]
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
        return tuple(order)

    def _require_active(self) -> None:
        """Reject work after this object has released its graph references."""
        if self._released:
            raise RuntimeError("Computation has been released")

    @property
    def nodes(self) -> list[Node]:
        """Return the cached dependency-first traversal as an independent list."""
        self._require_active()
        return list(self._nodes)

    def release(self) -> None:
        """Release graph references owned by this Computation.

        The output Variable remains usable if the caller retains it, but this
        Computation object cannot be replayed or differentiated afterwards.
        Calling ``release`` more than once is safe.
        """
        if self._released:
            return
        self.output = None
        self._nodes = ()
        self._released = True

    def forward(self) -> Tensor:
        """Recompute the output from its current leaf values."""
        self._require_active()
        values: dict[Any, Tensor] = {}
        for node in self._nodes:
            if node.op_cls is None:
                if node.output_var is None:
                    raise RuntimeError(
                        f"Leaf node {node!r} has no output Variable"
                    )
                values[node.output_var] = node.output_var.data
                continue

            result = self._execute_node(node, values)
            node.output_var.data = result
            node.capture_states()
            values[node.output_var] = result
        return values[self.output]

    def _validate_recorded_states(self) -> None:
        """Reject a backward pass whose recorded forward values changed."""
        self._require_active()
        for node in self._nodes:
            if node.op_cls is None:
                continue
            if node.output_changed():
                operation = node.label or getattr(
                    node.op_cls,
                    "__name__",
                    "operation",
                )
                raise RuntimeError(
                    f"Output of operation {operation!r} was modified after its "
                    "forward pass. Run a fresh forward pass or call "
                    "Computation(output).forward() before differentiation."
                )
            changed = node.changed_input()
            if changed is None:
                continue
            index, variable = changed
            operation = node.label or getattr(node.op_cls, "__name__", "operation")
            variable_name = getattr(variable, "name", None)
            description = (
                f" ({variable_name!r})" if variable_name is not None else ""
            )
            raise RuntimeError(
                f"Input {index}{description} to operation {operation!r} was "
                "modified after its forward pass. Run a fresh forward pass or "
                "call Computation(output).forward() before differentiation."
            )

    def backward(
        self,
        grad: TensorLike | None = None,
        *,
        create_graph: bool = False,
    ) -> None:
        """Differentiate the output with respect to reachable Variables."""
        if not isinstance(create_graph, bool):
            raise TypeError("create_graph must be a bool")
        self._validate_recorded_states()
        if create_graph:
            seed = self._gradient_seed(grad, create_graph=True)
            gradients = self._backward_graph(seed)
        else:
            seed = self._gradient_seed(grad)
            gradients = self._backward_values(seed)

        # Publish gradients only after the entire reverse pass succeeds. A
        # malformed operation or domain error therefore cannot leave a graph
        # with partially cleared or partially updated ``.grad`` attributes.
        for node in self._nodes:
            variable = node.output_var
            if variable is not None:
                variable.grad = gradients.get(variable)

    def _gradient_seed(
        self,
        grad: TensorLike | None,
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
        elif isinstance(grad, (int, float)) and self.output.data.shape == ():
            seed = Tensor([grad], shape=())
        else:
            seed = grad if isinstance(grad, Tensor) else Tensor(grad)

        seed_shape = seed.shape if isinstance(seed, Variable) else seed.shape
        if seed_shape != self.output.data.shape:
            raise ValueError(
                f"Gradient shape {seed_shape} does not match output shape "
                f"{self.output.data.shape}"
            )

        output_dtype = (
            self.output.dtype
            if self.output.dtype.typecode in {"f", "d"}
            else None
        )
        if output_dtype is not None and seed.dtype != output_dtype:
            if isinstance(seed, Variable):
                if create_graph and seed.requires_grad:
                    raise TypeError(
                        "A differentiable gradient seed must have the same "
                        "dtype as the output"
                    )
                seed = Variable(
                    seed.data.astype(output_dtype),
                    requires_grad=False,
                )
            else:
                seed = seed.astype(output_dtype)

        if create_graph:
            return seed if isinstance(seed, Variable) else Variable(seed, requires_grad=False)
        return seed.data if isinstance(seed, Variable) else seed

    def _backward_graph(self, seed: Any) -> dict[Any, Any]:
        """Build a differentiable reverse-mode gradient computation."""
        if not self.output.requires_grad:
            return {}
        gradient_terms: dict[Any, list[Any]] = {self.output: [seed]}
        gradients: dict[Any, Any] = {}
        for node in reversed(self._nodes):
            if node.op_cls is None:
                continue

            output = node.output_var
            output_terms = gradient_terms.get(output)
            if output_terms is None:
                continue
            output_gradient = self._sum_gradient_graph(output_terms)
            gradients[output] = output_gradient
            inputs = [edge.source.output_var for edge in node._in_edges]
            if not any(input_variable.requires_grad for input_variable in inputs):
                continue
            if not isinstance(node.op_cls, HigherOrderOperation):
                raise NotImplementedError(
                    f"Higher-order derivatives are not implemented for {node.label}"
                )
            input_gradients = node.op_cls.backward_graph(
                output_gradient,
                *inputs,
                **node.args,
            )
            input_gradients = self._validate_gradients(
                node, input_gradients, graph=True
            )
            for input_variable, input_gradient in zip(inputs, input_gradients):
                if not input_variable.requires_grad:
                    continue
                gradient_terms.setdefault(input_variable, []).append(input_gradient)

        for variable, terms in gradient_terms.items():
            if variable not in gradients:
                gradients[variable] = self._sum_gradient_graph(terms)
        return gradients

    def _backward_values(self, seed: Tensor) -> dict[Any, Tensor]:
        """Return numerical reverse-mode gradients without mutating Variables."""
        if not self.output.requires_grad:
            return {}
        gradient_terms: dict[Any, list[Tensor]] = {self.output: [seed]}
        gradients: dict[Any, Tensor] = {}
        for node in reversed(self._nodes):
            if node.op_cls is None:
                continue

            output = node.output_var
            output_terms = gradient_terms.get(output)
            if output_terms is None:
                continue
            output_gradient = self._sum_gradient_values(output_terms)
            gradients[output] = output_gradient

            input_data = [edge.source.output_var.data for edge in node._in_edges]
            input_gradients = node.op_cls.backward(
                output_gradient,
                *input_data,
                **node.args,
            )
            input_gradients = self._validate_gradients(
                node,
                input_gradients,
                graph=False,
            )
            for edge, input_gradient in zip(node._in_edges, input_gradients):
                input_variable = edge.source.output_var
                if not input_variable.requires_grad:
                    continue
                gradient_terms.setdefault(input_variable, []).append(input_gradient)

        for variable, terms in gradient_terms.items():
            if variable not in gradients:
                gradients[variable] = self._sum_gradient_values(terms)
        return gradients

    @staticmethod
    def _sum_gradient_values(gradients: list[Tensor]) -> Tensor:
        """Combine gradient contributions without order-dependent overflow."""
        if len(gradients) == 1:
            return gradients[0]

        from ..math.sum import _stable_float_sum

        first = gradients[0]
        values = [
            _stable_float_sum([
                float(gradient._data[index]) for gradient in gradients
            ])
            for index in range(first.size)
        ]
        return Tensor(values, dtype=first.dtype, shape=first.shape)

    @staticmethod
    def _sum_gradient_graph(gradients: list[Any]) -> Any:
        """Differentiably combine gradient contributions with stable summation."""
        if len(gradients) == 1:
            return gradients[0]

        from ..math import stack, sum

        return sum(stack(gradients, axis=0), axis=0)

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
        validated = list(results)
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
            input_variable = edge.source.output_var
            if (
                input_variable.requires_grad
                and gradient.dtype != input_variable.dtype
            ):
                validated[index] = gradient.astype(input_variable.dtype)
        return tuple(validated)

    @staticmethod
    def _execute_node(node: Node, values: dict[Any, Tensor]) -> Tensor:
        inputs = [edge.source.output_var for edge in node._in_edges]
        args = [values[value] for value in inputs]

        if node._scalar_operand:
            scalar = node.args["scalar"]
            if node.args.get("reverse", False):
                if not isinstance(node.op_cls, ReverseOperation):
                    raise TypeError(
                        f"{node.label} does not support a scalar left operand"
                    )
                result = node.op_cls.forward_reverse(args[0], scalar)
            else:
                result = node.op_cls.forward(args[0], scalar)
        else:
            result = node.op_cls.forward(*args, **node.args)

        return result if isinstance(result, Tensor) else Tensor([result])

def backward(
    output: Variable,
    grad: TensorLike | None = None,
    *,
    create_graph: bool = False,
) -> None:
    """Differentiate an output through its recorded Computation."""
    if not isinstance(create_graph, bool):
        raise TypeError("create_graph must be a bool")
    Computation(output).backward(grad, create_graph=create_graph)


@overload
def grad(
    output: Variable,
    inputs: Variable,
    grad_outputs: TensorLike | None = None,
    *,
    create_graph: bool = False,
) -> Tensor | Variable | None:
    ...


@overload
def grad(
    output: Variable,
    inputs: Iterable[Variable],
    grad_outputs: TensorLike | None = None,
    *,
    create_graph: bool = False,
) -> tuple[Tensor | Variable | None, ...]:
    ...


def grad(
    output: Variable,
    inputs: Variable | Iterable[Variable],
    grad_outputs: TensorLike | None = None,
    *,
    create_graph: bool = False,
) -> Tensor | Variable | None | tuple[Tensor | Variable | None, ...]:
    """Return gradients of ``output`` without modifying any ``.grad`` field.

    Set ``create_graph=True`` when the returned gradients will themselves be
    differentiated.
    """
    from ..variable import Variable

    if not isinstance(create_graph, bool):
        raise TypeError("create_graph must be a bool")

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
    computation._validate_recorded_states()
    if create_graph:
        seed = computation._gradient_seed(grad_outputs, create_graph=True)
        gradients = computation._backward_graph(seed)
        result = tuple(gradients.get(variable) for variable in requested)
    else:
        seed = computation._gradient_seed(grad_outputs)
        gradients = computation._backward_values(seed)
        result = tuple(gradients.get(variable) for variable in requested)
    return result[0] if single_input else result


__all__ = ["Computation", "backward", "grad"]

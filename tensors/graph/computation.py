"""A recorded computation rooted at an output Variable."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
import math
import threading
from typing import TYPE_CHECKING, Any, overload

from .._typing import TensorLike
from ..tensor import Tensor
from .node import Node

if TYPE_CHECKING:
    from ..variable import Variable


@dataclass(frozen=True)
class _ForwardInstruction:
    """One pre-resolved operation in a computation replay plan."""

    node: Node
    output_slot: int
    output_variable: Any
    input_slots: tuple[int, ...]
    input_variables: tuple[Any, ...]
    forward: Any
    reverse_forward: Any | None
    backward: Any
    backward_graph: Any | None
    arguments: dict[str, Any]
    scalar_operand: bool
    scalar: Any
    reverse: bool


@dataclass(frozen=True)
class _ForwardGroup:
    """One instruction or a CUDA-fusible scalar elementwise chain."""

    instructions: tuple[_ForwardInstruction, ...]
    fused_steps: tuple[tuple[str, float | None, bool], ...] | None = None


@dataclass
class _ExecutionWorkspace:
    """Thread-local reusable slot buffers for forward and backward execution."""

    values: list[Tensor | None]
    gradient_terms: list[list[Any]]
    gradients: list[Any | None]


class Computation:
    """A concrete computation that owns its forward and backward passes."""

    def __init__(self, output: Variable) -> None:
        if getattr(output, "node", None) is None:
            raise TypeError("Computation output must have a graph node")
        self.output = output
        self._nodes = self._dependency_order(output.node)
        self._compile_execution_plan()
        self._workspace_state = threading.local()
        self._released = False

    def _compile_execution_plan(self) -> None:
        """Resolve graph edges and operation methods once at construction."""
        variables = tuple(node.output_var for node in self._nodes)
        slots = {variable: index for index, variable in enumerate(variables)}
        instructions: list[_ForwardInstruction] = []
        leaf_slots: list[tuple[int, Any]] = []
        for node in self._nodes:
            output = node.output_var
            output_slot = slots[output]
            if node.op_cls is None:
                leaf_slots.append((output_slot, output))
                continue
            inputs = tuple(edge.source.output_var for edge in node._in_edges)
            reverse_forward = getattr(node.op_cls, "forward_reverse", None)
            instructions.append(
                _ForwardInstruction(
                    node=node,
                    output_slot=output_slot,
                    output_variable=output,
                    input_slots=tuple(slots[value] for value in inputs),
                    input_variables=inputs,
                    forward=node.op_cls.forward,
                    reverse_forward=reverse_forward,
                    backward=node.op_cls.backward,
                    backward_graph=getattr(
                        node.op_cls,
                        "backward_graph",
                        None,
                    ),
                    arguments=node.args,
                    scalar_operand=node._scalar_operand,
                    scalar=node.args.get("scalar"),
                    reverse=bool(node.args.get("reverse", False)),
                )
            )

        self._variables = variables
        self._leaf_slots = tuple(leaf_slots)
        self._forward_instructions = tuple(instructions)
        consumer_counts = [0] * len(variables)
        for instruction in instructions:
            for slot in instruction.input_slots:
                consumer_counts[slot] += 1
        self._forward_plan = self._group_forward_instructions(
            instructions,
            consumer_counts,
        )
        self._backward_plan = tuple(reversed(instructions))
        self._output_slot = slots[self.output]

    @classmethod
    def _group_forward_instructions(
        cls,
        instructions: list[_ForwardInstruction],
        consumer_counts: list[int],
    ) -> tuple[_ForwardGroup, ...]:
        """Collect linear, same-dtype scalar chains for optional CUDA fusion."""
        groups: list[_ForwardGroup] = []
        index = 0
        while index < len(instructions):
            first = instructions[index]
            first_step = cls._fused_scalar_step(first)
            if first_step is None:
                groups.append(_ForwardGroup((first,)))
                index += 1
                continue

            chain = [first]
            steps = [first_step]
            next_index = index + 1
            while next_index < len(instructions):
                candidate = instructions[next_index]
                step = cls._fused_scalar_step(candidate)
                if (
                    step is None
                    or candidate.input_slots != (chain[-1].output_slot,)
                    or consumer_counts[chain[-1].output_slot] != 1
                    or candidate.output_variable.shape
                    != first.output_variable.shape
                    or candidate.output_variable.dtype
                    != first.output_variable.dtype
                ):
                    break
                chain.append(candidate)
                steps.append(step)
                next_index += 1

            if len(chain) > 1:
                groups.append(
                    _ForwardGroup(tuple(chain), tuple(steps))
                )
                index = next_index
            else:
                groups.append(_ForwardGroup((first,)))
                index += 1
        return tuple(groups)

    @staticmethod
    def _fused_scalar_step(
        instruction: _ForwardInstruction,
    ) -> tuple[str, float | None, bool] | None:
        """Describe a scalar operation supported by the fused CUDA kernel."""
        names = {
            "Add": "add",
            "Sub": "subtract",
            "Mul": "multiply",
            "Div": "divide",
            "Neg": "negate",
        }
        operation = names.get(instruction.node.op_cls.__name__)
        if operation is None or len(instruction.input_variables) != 1:
            return None
        input_variable = instruction.input_variables[0]
        output_variable = instruction.output_variable
        if (
            input_variable.shape != output_variable.shape
            or input_variable.dtype != output_variable.dtype
            or output_variable.dtype.name != "float64"
        ):
            return None
        if operation == "negate":
            if instruction.scalar_operand:
                return None
            return operation, None, False
        if not instruction.scalar_operand or instruction.reverse:
            return None
        scalar = instruction.scalar
        if (
            isinstance(scalar, bool)
            or not isinstance(scalar, (int, float))
            or not math.isfinite(float(scalar))
            or (operation == "divide" and scalar == 0)
        ):
            return None
        return operation, float(scalar), False

    def _workspace(self) -> _ExecutionWorkspace:
        """Return reusable execution slots private to the current thread."""
        workspace = getattr(self._workspace_state, "workspace", None)
        if workspace is None or len(workspace.values) != len(self._variables):
            workspace = _ExecutionWorkspace(
                values=[None] * len(self._variables),
                gradient_terms=[[] for _ in self._variables],
                gradients=[None] * len(self._variables),
            )
            self._workspace_state.workspace = workspace
        return workspace

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
        self._variables = ()
        self._leaf_slots = ()
        self._forward_instructions = ()
        self._forward_plan = ()
        self._backward_plan = ()
        self._workspace_state.workspace = None
        self._released = True

    def forward(self) -> Tensor:
        """Recompute the output from its current leaf values."""
        self._require_active()
        values = self._workspace().values
        for slot, variable in self._leaf_slots:
            if variable is None:
                raise RuntimeError("A leaf node has no output Variable")
            values[slot] = variable.data

        for group in self._forward_plan:
            if group.fused_steps is not None and self._execute_fused_group(
                group,
                values,
            ):
                continue
            for instruction in group.instructions:
                self._execute_instruction(instruction, values)

        result = values[self._output_slot]
        if result is None:
            raise RuntimeError("Computation replay did not produce an output")
        return result

    @staticmethod
    def _execute_instruction(
        instruction: _ForwardInstruction,
        values: list[Tensor | None],
    ) -> None:
        """Execute one pre-resolved forward instruction."""
        args = []
        for slot in instruction.input_slots:
            value = values[slot]
            if value is None:
                raise RuntimeError("Computation input slot is uninitialized")
            args.append(value)

        if instruction.scalar_operand:
            if instruction.reverse:
                if instruction.reverse_forward is None:
                    raise TypeError(
                        f"{instruction.node.label} does not support a scalar "
                        "left operand"
                    )
                result = instruction.reverse_forward(
                    args[0],
                    instruction.scalar,
                )
            else:
                result = instruction.forward(args[0], instruction.scalar)
        else:
            result = instruction.forward(*args, **instruction.arguments)

        tensor = result if isinstance(result, Tensor) else Tensor([result])
        instruction.output_variable.data = tensor
        instruction.node.capture_states()
        values[instruction.output_slot] = tensor

    @staticmethod
    def _execute_fused_group(
        group: _ForwardGroup,
        values: list[Tensor | None],
    ) -> bool:
        """Execute a scalar chain in one CUDA kernel when it remains compatible."""
        from ..backend import execute_fused_elementwise

        first = group.instructions[0]
        value = values[first.input_slots[0]]
        if (
            value is None
            or value.shape != first.output_variable.shape
            or value.dtype != first.output_variable.dtype
        ):
            return False
        storages = execute_fused_elementwise(
            value,
            group.fused_steps or (),
            dtype=value.dtype,
        )
        if storages is None:
            return False
        if len(storages) != len(group.instructions):
            raise RuntimeError(
                "Fused elementwise kernel returned an unexpected output count"
            )
        for instruction, storage in zip(group.instructions, storages):
            tensor = Tensor(
                storage,
                dtype=instruction.output_variable.dtype,
                shape=value.shape,
            )
            instruction.output_variable.data = tensor
            instruction.node.capture_states()
            values[instruction.output_slot] = tensor
        return True

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
            from ..creation import ones

            typecode = (
                self.output.dtype.typecode
                if self.output.dtype.typecode in {"f", "d"}
                else "d"
            )
            seed = ones(
                self.output.data.shape,
                dtype=typecode,
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
        workspace = self._workspace()
        gradient_terms = workspace.gradient_terms
        gradients = workspace.gradients
        for terms in gradient_terms:
            terms.clear()
        for index in range(len(gradients)):
            gradients[index] = None
        gradient_terms[self._output_slot].append(seed)

        for instruction in self._backward_plan:
            output_terms = gradient_terms[instruction.output_slot]
            if not output_terms:
                continue
            output_gradient = self._sum_gradient_graph(output_terms)
            gradients[instruction.output_slot] = output_gradient
            inputs = instruction.input_variables
            if not any(variable.requires_grad for variable in inputs):
                continue
            if instruction.backward_graph is None:
                raise NotImplementedError(
                    "Higher-order derivatives are not implemented for "
                    f"{instruction.node.label}"
                )
            input_gradients = instruction.backward_graph(
                output_gradient,
                *inputs,
                **instruction.arguments,
            )
            input_gradients = self._validate_gradients(
                instruction.node,
                input_gradients,
                graph=True,
            )
            for slot, input_variable, input_gradient in zip(
                instruction.input_slots,
                inputs,
                input_gradients,
            ):
                if not input_variable.requires_grad:
                    continue
                gradient_terms[slot].append(input_gradient)

        for slot, terms in enumerate(gradient_terms):
            if terms and gradients[slot] is None:
                gradients[slot] = self._sum_gradient_graph(terms)
        return {
            variable: gradient
            for variable, gradient in zip(self._variables, gradients)
            if gradient is not None
        }

    def _backward_values(self, seed: Tensor) -> dict[Any, Tensor]:
        """Return numerical reverse-mode gradients without mutating Variables."""
        if not self.output.requires_grad:
            return {}
        workspace = self._workspace()
        gradient_terms = workspace.gradient_terms
        gradients = workspace.gradients
        for terms in gradient_terms:
            terms.clear()
        for index in range(len(gradients)):
            gradients[index] = None
        gradient_terms[self._output_slot].append(seed)

        for group in reversed(self._forward_plan):
            if group.fused_steps is not None and self._execute_fused_backward_group(
                group,
                gradient_terms,
                gradients,
            ):
                continue
            for instruction in reversed(group.instructions):
                self._execute_backward_instruction(
                    instruction,
                    gradient_terms,
                    gradients,
                )

        for slot, terms in enumerate(gradient_terms):
            if terms and gradients[slot] is None:
                gradients[slot] = self._sum_gradient_values(terms)
        return {
            variable: gradient
            for variable, gradient in zip(self._variables, gradients)
            if isinstance(gradient, Tensor)
        }

    def _execute_backward_instruction(
        self,
        instruction: _ForwardInstruction,
        gradient_terms: list[list[Any]],
        gradients: list[Any | None],
    ) -> None:
        """Execute one pre-resolved numerical VJP instruction."""
        output_terms = gradient_terms[instruction.output_slot]
        if not output_terms:
            return
        output_gradient = self._sum_gradient_values(output_terms)
        gradients[instruction.output_slot] = output_gradient
        input_data = tuple(
            variable.data for variable in instruction.input_variables
        )
        input_gradients = instruction.backward(
            output_gradient,
            *input_data,
            **instruction.arguments,
        )
        input_gradients = self._validate_gradients(
            instruction.node,
            input_gradients,
            graph=False,
        )
        for slot, input_variable, input_gradient in zip(
            instruction.input_slots,
            instruction.input_variables,
            input_gradients,
        ):
            if input_variable.requires_grad:
                gradient_terms[slot].append(input_gradient)

    def _execute_fused_backward_group(
        self,
        group: _ForwardGroup,
        gradient_terms: list[list[Any]],
        gradients: list[Any | None],
    ) -> bool:
        """Run a single-consumer scalar-chain VJP in one CUDA kernel."""
        from ..backend import execute_fused_elementwise

        reverse_instructions = tuple(reversed(group.instructions))
        last = reverse_instructions[0]
        output_terms = gradient_terms[last.output_slot]
        if not output_terms:
            return True
        output_gradient = self._sum_gradient_values(output_terms)
        gradients[last.output_slot] = output_gradient
        if not last.input_variables[0].requires_grad:
            return True

        derivative_steps = []
        for operation, scalar, reverse in reversed(group.fused_steps or ()):
            if reverse:
                return False
            if operation in {"add", "subtract"}:
                derivative_steps.append(("identity", None, False))
            elif operation == "multiply":
                derivative_steps.append(("multiply", scalar, False))
            elif operation == "divide":
                derivative_steps.append(("divide", scalar, False))
            elif operation == "negate":
                derivative_steps.append(("negate", None, False))
            else:
                return False

        storages = execute_fused_elementwise(
            output_gradient,
            tuple(derivative_steps),
            dtype=output_gradient.dtype,
        )
        if storages is None:
            return False
        current_output_gradient = output_gradient
        for instruction, storage in zip(reverse_instructions, storages):
            gradients[instruction.output_slot] = current_output_gradient
            input_variable = instruction.input_variables[0]
            input_gradient = Tensor(
                storage,
                dtype=input_variable.dtype,
                shape=input_variable.shape,
            )
            if input_variable.requires_grad:
                gradient_terms[instruction.input_slots[0]].append(
                    input_gradient
                )
            current_output_gradient = input_gradient
        return True

    @staticmethod
    def _sum_gradient_values(gradients: list[Tensor]) -> Tensor:
        """Combine gradient contributions without order-dependent overflow."""
        if len(gradients) == 1:
            return gradients[0]

        from ..backend import get_backend

        first = gradients[0]
        if get_backend() != "python" and first.size >= 32:
            from ..math import stack
            from ..math import sum as tensor_sum

            return tensor_sum(stack(gradients, axis=0), axis=0)

        from ..math.sum import _stable_float_sum

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

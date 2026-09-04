"""A recorded computation rooted at an output Variable."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, overload

from .._typing import TensorLike
from ..tensor import Tensor
from .fusion import execute_fused_backward, execute_fused_forward, plan_fusions
from .node import Node, VariableNode
from ..ops.operation import Operation

if TYPE_CHECKING:
    from ..variable import Variable


@dataclass(frozen=True, slots=True)
class Instruction:
    """One executable operation invocation.

    An instruction says which :class:`Operation` runs, which slots hold its
    operands, and which slot receives its result. Everything else about the
    invocation — the Variables themselves, their mutation records, and which
    of its VJPs a reverse pass wants — is resolved by the
    :class:`Computation` that owns the instruction.
    """

    operation: Operation
    input_slots: tuple[int, ...]
    output_slot: int


class Computation:
    """The planned, executable representation of a computational graph.

    Constructing a Computation inspects the graph rooted at an output
    Variable, resolves it into ordered :class:`Instruction` objects over
    numbered Variable slots, and retains that plan for replay and
    differentiation. The Computation is the plan: :meth:`forward` and
    :meth:`backward` execute it.
    """

    def __init__(self, output: Variable) -> None:
        outputs = self._validate_outputs((output,))
        nodes, node_masks, boundary_nodes = self._dependency_plan(outputs, ())
        self._initialize_plan(
            outputs[0],
            nodes,
            node_masks,
            boundary_nodes,
            1,
        )

    @classmethod
    def _for_autograd(cls, output: Variable) -> Computation:
        """Return the reusable reverse plan owned by an output Variable.

        The plan is execution state, so it lives on the runtime value rather
        than on a structural graph node. Graph topology is immutable, so later
        calls reuse the pre-resolved plan; each pass allocates its own
        execution buffers and shares none of them.
        """
        cls._validate_outputs((output,))
        computation = output._autograd_computation
        if computation is None or computation._released:
            computation = cls(output)
            output._autograd_computation = computation
        return computation

    @classmethod
    def from_outputs(
        cls,
        outputs: Iterable[Variable],
        *,
        boundaries: Iterable[Variable] = (),
    ) -> tuple[Computation, ...]:
        """Build output views over one shared multi-root execution plan."""
        output_tuple = cls._validate_outputs(tuple(outputs))
        boundary_tuple = tuple(boundaries)
        nodes, node_masks, boundary_nodes = cls._dependency_plan(
            output_tuple,
            boundary_tuple,
        )

        first = cls.__new__(cls)
        first._initialize_plan(
            output_tuple[0],
            nodes,
            node_masks,
            boundary_nodes,
            1,
        )
        computations = [first]
        for index, output in enumerate(output_tuple[1:], 1):
            computation = cls.__new__(cls)
            computation._initialize_shared_view(output, first, 1 << index)
            computations.append(computation)
        return tuple(computations)

    @staticmethod
    def _validate_outputs(outputs: tuple[Variable, ...]) -> tuple[Variable, ...]:
        if not outputs:
            raise ValueError("Computation requires at least one output")
        for output in outputs:
            if not isinstance(getattr(output, "node", None), VariableNode):
                raise TypeError("Computation output must have a graph node")
        return outputs

    def _initialize_plan(
        self,
        output: Variable,
        nodes: tuple[Node, ...],
        node_masks: tuple[int, ...],
        boundary_nodes: frozenset[Node],
        output_bit: int,
    ) -> None:
        """Initialize the owner of a new shared execution plan."""
        self.output = output
        self._all_nodes = nodes
        self._node_masks = node_masks
        self._boundary_nodes = boundary_nodes
        self._output_bit = output_bit
        self._nodes = tuple(
            node
            for node, mask in zip(nodes, node_masks)
            if mask & output_bit
        )
        self._compile_execution_plan()
        self._select_view()
        self._released = False

    def _initialize_shared_view(
        self,
        output: Variable,
        owner: Computation,
        output_bit: int,
    ) -> None:
        """Initialize another output view without rebuilding the shared plan."""
        self.output = output
        self._all_nodes = owner._all_nodes
        self._node_masks = owner._node_masks
        self._boundary_nodes = owner._boundary_nodes
        self._output_bit = output_bit
        self._nodes = tuple(
            node
            for node, mask in zip(self._all_nodes, self._node_masks)
            if mask & output_bit
        )
        self._variables = owner._variables
        self._variable_slots = owner._variable_slots
        self._leaf_slots = owner._leaf_slots
        self._instructions = owner._instructions
        self._fusions = owner._fusions
        self._fusion_starts = owner._fusion_starts
        self._output_slot = self._variable_slots[output]
        self._select_view()
        self._released = False

    def _select_view(self) -> None:
        """Resolve the slots and instructions this output reaches."""
        slots = self._variable_slots
        view_slots = {
            slots[node.variable]
            for node in self._nodes
            if isinstance(node, VariableNode)
        }
        self._view_slots = tuple(sorted(view_slots))
        self._view_instructions = tuple(
            instruction
            for instruction in self._instructions
            if instruction.output_slot in view_slots
        )

    def _compile_execution_plan(self) -> None:
        """Resolve graph relationships and operation methods once.

        Every Variable vertex becomes an execution slot. Every operation vertex
        becomes one instruction whose operands are named by its incoming edges
        and whose result is the Variable named by its outgoing edge. Replay and
        differentiation then work from this compact plan instead of walking the
        graph again.
        """
        variables = tuple(
            node.variable
            for node in self._all_nodes
            if isinstance(node, VariableNode)
        )
        slots = {variable: index for index, variable in enumerate(variables)}
        instructions: list[Instruction] = []
        leaf_slots: list[int] = []
        boundary_nodes = self._boundary_nodes
        for node in self._all_nodes:
            if not isinstance(node, VariableNode):
                continue
            output_slot = slots[node.variable]
            producer = node.producer
            if producer is None or node in boundary_nodes:
                leaf_slots.append(output_slot)
                continue
            instructions.append(
                Instruction(
                    operation=producer.operation,
                    input_slots=tuple(
                        slots[edge.source.variable]
                        for edge in producer._in_edges
                    ),
                    output_slot=output_slot,
                )
            )

        self._variables = variables
        self._variable_slots = slots
        self._leaf_slots = tuple(leaf_slots)
        self._instructions = tuple(instructions)
        self._output_slot = slots[self.output]
        # Fusion is an optional acceleration of this sequence, recorded beside
        # it. The instructions themselves stay the canonical plan.
        self._fusions, self._fusion_starts = plan_fusions(
            self._instructions,
            variables,
        )

    def _live_slots(self, targets: tuple[Variable, ...] | None) -> set[int]:
        """Return the slots whose gradient this reverse invocation requires.

        ``targets is None`` requests a gradient at every reachable
        differentiable Variable, which is what ``backward`` publishes. A
        target tuple instead requests only the reverse paths connecting the
        output to those Variables, so a VJP runs solely where a requested
        Variable's influence actually flows.

        The result is transient analysis data for one reverse call. It is
        never cached, because ``requires_grad`` may change between passes.
        """
        self._require_active()
        variables = self._variables
        if targets is None:
            return {
                slot
                for slot, variable in enumerate(variables)
                if variable.requires_grad
            }

        slots = self._variable_slots
        live = set()
        for variable in targets:
            slot = slots.get(variable)
            if slot is not None and variable.requires_grad:
                live.add(slot)
        # Instructions are in dependency order, so one forward sweep closes
        # the set over every path from a requested Variable to the output.
        for instruction in self._instructions:
            if variables[instruction.output_slot].requires_grad and any(
                slot in live for slot in instruction.input_slots
            ):
                live.add(instruction.output_slot)
        return live

    @classmethod
    def _dependency_plan(
        cls,
        outputs: tuple[Variable, ...],
        boundaries: tuple[Variable, ...],
    ) -> tuple[tuple[Node, ...], tuple[int, ...], frozenset[Node]]:
        """Return one traversal and per-output reachability masks."""
        boundary_nodes = frozenset(
            variable.node
            for variable in boundaries
            if getattr(variable, "node", None) is not None
        )
        order: list[Node] = []
        visited: set[Node] = set()
        for output in outputs:
            stack = [(output.node, False)]
            while stack:
                node, expanded = stack.pop()
                if expanded:
                    order.append(node)
                    continue
                if node in visited:
                    continue
                visited.add(node)
                stack.append((node, True))
                if node in boundary_nodes:
                    continue
                for edge in reversed(node._in_edges):
                    if edge.source not in visited:
                        stack.append((edge.source, False))

        masks = {node: 0 for node in order}
        for index, output in enumerate(outputs):
            masks[output.node] |= 1 << index
        for node in reversed(order):
            mask = masks[node]
            if not mask or node in boundary_nodes:
                continue
            for edge in node._in_edges:
                masks[edge.source] |= mask
        return tuple(order), tuple(masks[node] for node in order), boundary_nodes

    @classmethod
    def _dependency_order(cls, output_node: VariableNode) -> tuple[Node, ...]:
        """Calculate the dependency-first traversal reaching an output."""
        nodes, _, _ = cls._dependency_plan((output_node.variable,), ())
        return nodes

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
        self._all_nodes = ()
        self._node_masks = ()
        self._boundary_nodes = frozenset()
        self._nodes = ()
        self._variables = ()
        self._variable_slots = {}
        self._leaf_slots = ()
        self._instructions = ()
        self._fusions = {}
        self._fusion_starts = {}
        self._view_slots = ()
        self._view_instructions = ()
        self._released = True

    def forward(self) -> Tensor:
        """Recompute the output from its current leaf values."""
        self._require_active()
        variables = self._variables
        # Execution buffers are ordinary locals: every call owns its own, so
        # concurrent replays of one Computation never share mutable state.
        values: list[Tensor | None] = [None] * len(variables)
        for slot in self._leaf_slots:
            values[slot] = variables[slot].data

        instructions = self._instructions
        fusions = self._fusions
        index = 0
        count = len(instructions)
        while index < count:
            fusion = fusions.get(index)
            if fusion is not None and execute_fused_forward(
                index,
                fusion,
                instructions,
                variables,
                values,
            ):
                index = fusion[0] + 1
                continue
            self._execute_instruction(instructions[index], values)
            index += 1

        result = values[self._output_slot]
        if result is None:
            raise RuntimeError("Computation replay did not produce an output")
        return result

    def _execute_instruction(
        self,
        instruction: Instruction,
        values: list[Tensor | None],
    ) -> None:
        """Execute one instruction into its output slot."""
        input_slots = instruction.input_slots
        args = []
        for slot in input_slots:
            value = values[slot]
            if value is None:
                raise RuntimeError("Computation input slot is uninitialized")
            args.append(value)

        result = instruction.operation.forward(*args)

        tensor = result if isinstance(result, Tensor) else Tensor([result])
        variables = self._variables
        output = variables[instruction.output_slot]
        output._replace_data_from_replay(tensor)
        output._capture_forward_record(
            variables[slot] for slot in input_slots
        )
        values[instruction.output_slot] = tensor

    def _validate_recorded_states(self) -> None:
        """Reject a backward pass whose recorded forward values changed."""
        self._require_active()
        variables = self._variables
        for instruction in self._view_instructions:
            output = variables[instruction.output_slot]
            record = output._forward_record
            if record is None:
                continue
            input_states, output_state = record
            operation = instruction.operation.name
            if output._mutation_state() != output_state:
                raise RuntimeError(
                    f"Output of operation {operation!r} was modified after its "
                    "forward pass. Run a fresh forward pass or call "
                    "Computation(output).forward() before differentiation."
                )
            inputs = [variables[slot] for slot in instruction.input_slots]
            if len(input_states) != len(inputs):
                index, variable = 0, None
            else:
                for index, (variable, expected) in enumerate(
                    zip(inputs, input_states)
                ):
                    if variable._mutation_state() != expected:
                        break
                else:
                    continue
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
        live = self._live_slots(None)
        if create_graph:
            seed = self._gradient_seed(grad, create_graph=True)
            gradients = self._backward_graph(seed, live)
        else:
            seed = self._gradient_seed(grad)
            gradients = self._backward_values(seed, live)

        # Publish gradients only after the entire reverse pass succeeds. A
        # malformed operation or domain error therefore cannot leave a graph
        # with partially cleared or partially updated ``.grad`` attributes.
        variables = self._variables
        for slot in self._view_slots:
            variable = variables[slot]
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

    def _backward_graph(
        self,
        seed: Any,
        live: set[int],
    ) -> dict[Any, Any]:
        """Build a differentiable reverse-mode gradient computation."""
        if not self.output.requires_grad:
            return {}
        count = len(self._variables)
        gradient_terms: list[list[Any]] = [[] for _ in range(count)]
        gradients: list[Any | None] = [None] * count
        gradient_terms[self._output_slot].append(seed)
        variables = self._variables

        for instruction in reversed(self._instructions):
            output_terms = gradient_terms[instruction.output_slot]
            if not output_terms:
                continue
            output_gradient = self._sum_gradient_graph(output_terms)
            gradients[instruction.output_slot] = output_gradient
            input_slots = instruction.input_slots
            needs_input_grad = tuple(slot in live for slot in input_slots)
            if not any(needs_input_grad):
                continue
            inputs = tuple(variables[slot] for slot in input_slots)
            input_gradients = instruction.operation.backward_graph(
                output_gradient,
                *inputs,
                needs_input_grad=needs_input_grad,
            )
            input_gradients = self._validate_gradients(
                instruction.operation,
                inputs,
                input_gradients,
                needs_input_grad,
                graph=True,
            )
            for slot, wanted, input_gradient in zip(
                input_slots,
                needs_input_grad,
                input_gradients,
            ):
                if wanted:
                    gradient_terms[slot].append(input_gradient)

        for slot, terms in enumerate(gradient_terms):
            if terms and gradients[slot] is None:
                gradients[slot] = self._sum_gradient_graph(terms)
        return {
            variable: gradient
            for variable, gradient in zip(self._variables, gradients)
            if gradient is not None
        }

    def _backward_values(
        self,
        seed: Tensor,
        live: set[int],
    ) -> dict[Any, Tensor]:
        """Return numerical reverse-mode gradients without mutating Variables."""
        if not self.output.requires_grad:
            return {}
        count = len(self._variables)
        gradient_terms: list[list[Any]] = [[] for _ in range(count)]
        gradients: list[Any | None] = [None] * count
        gradient_terms[self._output_slot].append(seed)

        instructions = self._instructions
        fusion_starts = self._fusion_starts
        index = len(instructions) - 1
        while index >= 0:
            start = fusion_starts.get(index)
            if start is None:
                self._execute_backward_instruction(
                    instructions[index],
                    gradient_terms,
                    gradients,
                    live,
                )
                index -= 1
                continue
            if not execute_fused_backward(
                start,
                self._fusions[start],
                instructions,
                self._variables,
                gradient_terms,
                gradients,
                live,
            ):
                # The fused VJP cannot serve this demand, so the same
                # instructions run ordinarily in reverse.
                for position in range(index, start - 1, -1):
                    self._execute_backward_instruction(
                        instructions[position],
                        gradient_terms,
                        gradients,
                        live,
                    )
            index = start - 1

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
        instruction: Instruction,
        gradient_terms: list[list[Any]],
        gradients: list[Any | None],
        live: set[int],
    ) -> None:
        """Execute one instruction's requested numerical VJPs."""
        output_terms = gradient_terms[instruction.output_slot]
        if not output_terms:
            return
        output_gradient = self._sum_gradient_values(output_terms)
        gradients[instruction.output_slot] = output_gradient
        input_slots = instruction.input_slots
        needs_input_grad = tuple(slot in live for slot in input_slots)
        if not any(needs_input_grad):
            # Nothing this reverse pass wants lies behind this operation.
            return
        variables = self._variables
        inputs = tuple(variables[slot] for slot in input_slots)
        input_gradients = instruction.operation.backward(
            output_gradient,
            *(variable.data for variable in inputs),
            needs_input_grad=needs_input_grad,
        )
        input_gradients = self._validate_gradients(
            instruction.operation,
            inputs,
            input_gradients,
            needs_input_grad,
            graph=False,
        )
        for slot, wanted, input_gradient in zip(
            input_slots,
            needs_input_grad,
            input_gradients,
        ):
            if wanted:
                gradient_terms[slot].append(input_gradient)

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
    def _validate_gradients(
        operation: Operation,
        inputs: Sequence[Variable],
        gradients: Any,
        needs_input_grad: tuple[bool, ...],
        *,
        graph: bool,
    ) -> tuple[Any, ...]:
        """Validate an operation's VJP result against the requested demand.

        A requested input must receive a value of the expected type, shape,
        and dtype. An unrequested input must receive ``None``: returning a
        value there means the operation calculated work the reverse pass did
        not ask for, which makes the demand contract enforceable rather than
        advisory. A real zero remains a valid answer for a requested
        derivative whose value is mathematically zero.
        """
        from ..variable import Variable

        label = operation.name
        mode = "backward_graph" if graph else "backward"
        try:
            results = tuple(gradients)
        except TypeError as exc:
            raise TypeError(
                f"{label} backward must return one gradient per input"
            ) from exc

        if len(results) != len(inputs):
            raise RuntimeError(
                f"{label} backward returned {len(results)} gradients for "
                f"{len(inputs)} inputs"
            )

        expected_type = Variable if graph else Tensor
        validated = list(results)
        for index, (input_variable, gradient, wanted) in enumerate(
            zip(inputs, results, needs_input_grad)
        ):
            if not wanted:
                if gradient is not None:
                    raise RuntimeError(
                        f"{label} {mode} returned a gradient for input "
                        f"{index}, which this reverse pass did not request; "
                        "return None for an unrequested derivative"
                    )
                continue
            if gradient is None:
                raise RuntimeError(
                    f"{label} {mode} returned None for input {index}, whose "
                    "gradient this reverse pass requested"
                )
            if not isinstance(gradient, expected_type):
                raise TypeError(
                    f"{label} {mode} gradient {index} must be a "
                    f"{expected_type.__name__}, got {type(gradient).__name__}"
                )
            if gradient.shape != input_variable.shape:
                raise ValueError(
                    f"{label} backward gradient {index} has shape "
                    f"{gradient.shape}; expected {input_variable.shape}"
                )
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
    Computation._for_autograd(output).backward(
        grad,
        create_graph=create_graph,
    )


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
    computation = Computation._for_autograd(output)
    computation._validate_recorded_states()
    # Build the reverse demand from the requested inputs so only the paths
    # connecting them to the output are differentiated.
    live = computation._live_slots(requested)
    if create_graph:
        seed = computation._gradient_seed(grad_outputs, create_graph=True)
        gradients = computation._backward_graph(seed, live)
    else:
        seed = computation._gradient_seed(grad_outputs)
        gradients = computation._backward_values(seed, live)
    result = tuple(gradients.get(variable) for variable in requested)
    return result[0] if single_input else result


__all__ = ["Computation", "backward", "grad"]

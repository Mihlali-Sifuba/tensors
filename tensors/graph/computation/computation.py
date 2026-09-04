"""A recorded computation rooted at an output Variable."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ..._typing import TensorLike
from ...tensor import Tensor
from .fusion import execute_fused_backward, execute_fused_forward, plan_fusions
from .gradients import (
    gradient_seed,
    sum_gradient_graph,
    sum_gradient_values,
    validate_gradients,
)
from ..node import Node, VariableNode
from ...ops.operation import Operation

if TYPE_CHECKING:
    from ...variable import Variable


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
            seed = gradient_seed(self.output, grad, create_graph=True)
            gradients = self._backward_graph(seed, live)
        else:
            seed = gradient_seed(self.output, grad)
            gradients = self._backward_values(seed, live)

        # Publish gradients only after the entire reverse pass succeeds. A
        # malformed operation or domain error therefore cannot leave a graph
        # with partially cleared or partially updated ``.grad`` attributes.
        variables = self._variables
        for slot in self._view_slots:
            variable = variables[slot]
            variable.grad = gradients.get(variable)

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
            output_gradient = sum_gradient_graph(output_terms)
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
            input_gradients = validate_gradients(
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
                gradients[slot] = sum_gradient_graph(terms)
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
                gradients[slot] = sum_gradient_values(terms)
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
        output_gradient = sum_gradient_values(output_terms)
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
        input_gradients = validate_gradients(
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


__all__ = ["Computation"]

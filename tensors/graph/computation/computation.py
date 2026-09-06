"""Execution of a compiled computation over node-identified value slots."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

from ..._typing import TensorLike
from ...tensor import Tensor
from .compiler import Compiler, resolve_boundaries, resolve_outputs
from .fusion import execute_fused_backward, execute_fused_forward, plan_fusions
from .gradients import (
    gradient_seed,
    sum_gradient_graph,
    sum_gradient_values,
    validate_gradients,
)

if TYPE_CHECKING:
    from ...variable import Variable
    from ..node import Node, VariableNode
    from .fusion import Fusion
    from .instruction import Instruction


class _FusionPlan:
    """One compiled program's fusion plan, resolved once it can be.

    Fusion is an optional acceleration of the instruction sequence, recorded
    beside it; the instructions themselves stay the canonical program. A
    fusible run is recognized from the shapes and dtypes its slots hold, so
    the plan cannot be made before those values exist. Every view of one
    compilation shares this holder, so the plan is still resolved once.
    """

    __slots__ = ("_instructions", "_nodes", "_plan")

    def __init__(
        self,
        instructions: tuple[Instruction, ...],
        nodes: tuple[VariableNode, ...],
    ) -> None:
        self._instructions = instructions
        self._nodes = nodes
        self._plan: tuple[dict[int, Fusion], dict[int, int]] | None = None

    def resolve(self) -> tuple[dict[int, Fusion], dict[int, int]]:
        """Return the plan, or no fusion while a slot holds no value yet.

        Declining to fuse is not a fallback for a missing value: an
        unfusable program executes exactly the same instructions.
        """
        plan = self._plan
        if plan is None:
            if not all(node.is_bound for node in self._nodes):
                return {}, {}
            plan = plan_fusions(
                self._instructions,
                tuple(node.variable for node in self._nodes),
            )
            self._plan = plan
        return plan


#: The plan of a program with nothing left to execute.
_NO_FUSION = _FusionPlan((), ())


def _plan_fusion(compiler: Compiler) -> _FusionPlan:
    """Return the fusion plan every view of one compilation shares."""
    return _FusionPlan(compiler.instructions, compiler.variable_nodes)


class Computation:
    """The compiled, executable representation of a computational graph.

    A :class:`~tensors.graph.computation.compiler.Compiler` turns the graph
    rooted at an output Variable into ordered
    :class:`~tensors.graph.computation.instruction.Instruction` objects over
    numbered value slots, and resolves the part of that program each output
    needs. A Computation holds one such compiled program together with its
    own execution view and executes it: :meth:`forward` replays it and
    :meth:`backward` differentiates it.

    A slot is identified by the vertex naming its value and holds a Tensor
    while a pass runs, so execution does not presuppose the runtime Variables
    it produces. Running an instruction gives its output slot's vertex the
    Tensor it names: the first pass materializes the Variable that vertex was
    always going to have, and a replay updates the one already bound to it.
    Differentiation works on those Variables, so it requires a forward pass
    to have produced them.

    Everything here is expressed in the compiled domain — vertices, slots,
    values, instructions, and fusion metadata. The topology the program came
    from is the compiler's concern, not the Computation's.
    """

    def __init__(self, output: Variable) -> None:
        compiler = Compiler(resolve_outputs((output,)))
        compiler.compile()
        self._adopt_program(compiler, 0, _plan_fusion(compiler))

    @classmethod
    def from_nodes(
        cls,
        outputs: Iterable[VariableNode],
        *,
        boundaries: Iterable[VariableNode] = (),
    ) -> tuple[Computation, ...]:
        """Build output views over the program rooted at output vertices.

        This is the graph-first entry point: the vertices name the values the
        program produces, and none of them has to hold one yet.
        """
        compiler = Compiler(outputs, boundaries=boundaries)
        compiler.compile()
        return cls._from_compiler(compiler)

    @classmethod
    def from_outputs(
        cls,
        outputs: Iterable[Variable],
        *,
        boundaries: Iterable[Variable] = (),
    ) -> tuple[Computation, ...]:
        """Build output views over one shared multi-root compiled program."""
        compiler = Compiler(
            resolve_outputs(outputs),
            boundaries=resolve_boundaries(boundaries),
        )
        compiler.compile()
        return cls._from_compiler(compiler)

    @classmethod
    def _from_compiler(cls, compiler: Compiler) -> tuple[Computation, ...]:
        """Build one view per output of an already-compiled program.

        A caller that also wants the compiler's structural metadata compiles
        once and hands that compiler here, so one trace is never compiled
        twice.
        """
        fusion = _plan_fusion(compiler)
        computations = []
        for index in range(len(compiler.output_nodes)):
            computation = cls.__new__(cls)
            computation._adopt_program(compiler, index, fusion)
            computations.append(computation)
        return tuple(computations)

    def _adopt_program(
        self,
        compiler: Compiler,
        index: int,
        fusion: _FusionPlan,
    ) -> None:
        """Take one output's view of an already-compiled program.

        The program is adopted exactly as it was compiled — vertices, slots
        and instructions — so a view exists as soon as the graph does, before
        anything has produced the values it names. Every view of one
        compilation stores the same program objects, so multiple outputs
        share their instructions, slots and fusion metadata while each keeps
        its own view of them.
        """
        self._output_node = compiler.output_nodes[index]
        self._variable_nodes = compiler.variable_nodes
        self._node_slots = compiler.node_slots
        self._leaf_slots = compiler.leaf_slots
        self._instructions = compiler.instructions
        self._output_slot = compiler.output_slots[index]
        self._view_slots = compiler.view_slots[index]
        self._view_instructions = compiler.view_instructions[index]
        # Structural nodes are opaque here: they are carried for the public
        # ``nodes`` property and never interpreted.
        self._view_nodes = compiler.view_nodes[index]
        self._fusion = fusion
        # The runtime projection of the slot table, resolved once a pass
        # needs values rather than at construction.
        self._variables: tuple[Variable, ...] | None = None
        self._released = False

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
        variables = self._require_materialized()
        if targets is None:
            return {
                slot
                for slot, variable in enumerate(variables)
                if variable.requires_grad
            }

        # A Variable and the vertex naming it are one to one, so a requested
        # target names its own slot.
        slots = self._node_slots
        live = set()
        for variable in targets:
            slot = slots.get(variable.node)
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

    def _require_active(self) -> None:
        """Reject work after this object has released its graph references."""
        if self._released:
            raise RuntimeError("Computation has been released")

    def _require_materialized(self) -> tuple[Variable, ...]:
        """Return the Variable occupying each slot of this program.

        Differentiation and fused execution work on Variables, so they need
        every slot to hold one. A program whose graph named values nothing
        has produced yet has none until :meth:`forward` runs, and says so
        rather than differentiating an incomplete pass. Binding is permanent,
        so the resolved projection is kept once it is complete.
        """
        self._require_active()
        variables = self._variables
        if variables is None:
            nodes = self._variable_nodes
            for slot, node in enumerate(nodes):
                if not node.is_bound:
                    raise RuntimeError(
                        f"Computation slot {slot} holds no value yet. Run "
                        "forward() before differentiating this computation."
                    )
            variables = tuple(node.variable for node in nodes)
            self._variables = variables
        return variables

    def _leaf_variable(self, slot: int) -> Variable:
        """Return the Variable a leaf slot reads its value from.

        A leaf is a value the program consumes but never produces, so it must
        already exist: nothing downstream can supply it.
        """
        node = self._variable_nodes[slot]
        if not node.is_bound:
            raise RuntimeError(
                f"Computation leaf slot {slot} holds no value. The graph "
                "names it, but nothing has materialized it, and no "
                "instruction produces it."
            )
        return node.variable

    def _leaf_variables(self) -> tuple[Variable, ...]:
        """Return the Variables this program reads but never produces."""
        self._require_active()
        return tuple(self._leaf_variable(slot) for slot in self._leaf_slots)

    @property
    def output(self) -> Variable:
        """Return the runtime Variable this view produces.

        The output slot is named by a vertex from the moment the graph is
        compiled; it holds a Variable once something has materialized one.
        For a recorded eager graph that is already true, and for a graph
        compiled ahead of its values it becomes true when :meth:`forward`
        produces the output.
        """
        self._require_active()
        return self._output_node.variable

    @property
    def nodes(self) -> list[Node]:
        """Return the cached dependency-first traversal as an independent list.

        The compiler resolved this traversal for this output; a Computation
        only hands it back.
        """
        self._require_active()
        return list(self._view_nodes)

    @property
    def _fusions(self) -> dict[int, Fusion]:
        """Return this program's fusible instruction runs, by start index."""
        return self._fusion.resolve()[0]

    @property
    def _fusion_starts(self) -> dict[int, int]:
        """Return where each fusible run starts, keyed by where it ends."""
        return self._fusion.resolve()[1]

    def release(self) -> None:
        """Release graph references owned by this Computation.

        The output Variable remains usable if the caller retains it, but this
        Computation object cannot be replayed or differentiated afterwards.
        Calling ``release`` more than once is safe.
        """
        if self._released:
            return
        self._output_node = None
        self._view_nodes = ()
        self._variable_nodes = ()
        self._node_slots = {}
        self._variables = None
        self._leaf_slots = ()
        self._instructions = ()
        self._fusion = _NO_FUSION
        self._view_slots = ()
        self._view_instructions = ()
        self._released = True

    def forward(self) -> Tensor:
        """Produce the output from this program's current leaf values.

        The pass runs over Tensors held in slots. Each instruction gives its
        result to the vertex naming that slot, which materializes a Variable
        the first time and updates the bound one afterwards, so a graph
        compiled before its values exist runs the same program as a replay of
        a recorded one.
        """
        self._require_active()
        # Execution buffers are ordinary locals: every call owns its own, so
        # concurrent replays of one Computation never share mutable state.
        values: list[Tensor | None] = [None] * len(self._variable_nodes)
        for slot in self._leaf_slots:
            values[slot] = self._leaf_variable(slot).data

        instructions = self._instructions
        fusions, _ = self._fusion.resolve()
        # A fused run reads the shapes and dtypes of the slots it spans, so a
        # plan exists only once every slot holds a value.
        variables = self._require_materialized() if fusions else ()
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
        values[instruction.output_slot] = tensor
        self._bind_result(instruction, tensor)

    def _bind_result(self, instruction: Instruction, tensor: Tensor) -> None:
        """Give an instruction's output vertex the value it just produced.

        An unbound vertex is materialized as the Variable it always named; a
        bound one keeps its Variable and takes the new value, so replay never
        binds a second Variable to a vertex. Either way the forward state is
        captured afterwards, from Variables that now all exist.
        """
        nodes = self._variable_nodes
        input_slots = instruction.input_slots
        node = nodes[instruction.output_slot]
        if node.is_bound:
            output = node.variable
            output._replace_data_from_replay(tensor)
        else:
            # A result is differentiable exactly when an operand is, which is
            # the rule an eagerly recorded result follows too.
            output = node.materialize(
                tensor,
                requires_grad=any(
                    nodes[slot].variable.requires_grad for slot in input_slots
                ),
            )
        output._capture_forward_state(
            nodes[slot].variable for slot in input_slots
        )

    def _validate_recorded_states(self) -> None:
        """Reject a backward pass whose recorded forward values changed."""
        variables = self._require_materialized()
        for instruction in self._view_instructions:
            output = variables[instruction.output_slot]
            record = output._forward_state
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
        variables = self._require_materialized()
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
        variables = self._require_materialized()
        count = len(variables)
        gradient_terms: list[list[Any]] = [[] for _ in range(count)]
        gradients: list[Any | None] = [None] * count
        gradient_terms[self._output_slot].append(seed)

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
            for variable, gradient in zip(variables, gradients)
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
        variables = self._require_materialized()
        count = len(variables)
        gradient_terms: list[list[Any]] = [[] for _ in range(count)]
        gradients: list[Any | None] = [None] * count
        gradient_terms[self._output_slot].append(seed)

        instructions = self._instructions
        fusions, fusion_starts = self._fusion.resolve()
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
                fusions[start],
                instructions,
                variables,
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
            for variable, gradient in zip(variables, gradients)
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
        variables = self._require_materialized()
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

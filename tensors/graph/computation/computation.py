"""Execution of a compiled computation rooted at an output Variable."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

from ..._typing import TensorLike
from ...tensor import Tensor
from .compiler import Compiler
from .fusion import execute_fused_backward, execute_fused_forward, plan_fusions
from .gradients import (
    gradient_seed,
    sum_gradient_graph,
    sum_gradient_values,
    validate_gradients,
)

if TYPE_CHECKING:
    from ...variable import Variable
    from ..node import Node
    from .fusion import Fusion
    from .instruction import Instruction


def _fusion_plan(
    compiler: Compiler,
) -> tuple[dict[int, Fusion], dict[int, int]]:
    """Plan fusion once over a compiled program; every view shares the plan.

    Fusion is an optional acceleration of the instruction sequence, recorded
    beside it. The instructions themselves stay the canonical program.
    """
    return plan_fusions(compiler.instructions, compiler.variables)


class Computation:
    """The compiled, executable representation of a computational graph.

    A :class:`~tensors.graph.computation.compiler.Compiler` turns the graph
    rooted at an output Variable into ordered
    :class:`~tensors.graph.computation.instruction.Instruction` objects over
    numbered Variable slots, and resolves the part of that program each
    output needs. A Computation holds one such compiled program together with
    its own execution view and executes it: :meth:`forward` replays it and
    :meth:`backward` differentiates it.

    Everything here is expressed in the compiled domain — Variables, slots,
    instructions, and fusion metadata. The graph the program came from is the
    compiler's concern, not the Computation's.
    """

    def __init__(self, output: Variable) -> None:
        compiler = Compiler((output,))
        compiler.compile()
        self._adopt_program(compiler, 0, _fusion_plan(compiler))

    @classmethod
    def from_outputs(
        cls,
        outputs: Iterable[Variable],
        *,
        boundaries: Iterable[Variable] = (),
    ) -> tuple[Computation, ...]:
        """Build output views over one shared multi-root compiled program."""
        compiler = Compiler(outputs, boundaries=boundaries)
        compiler.compile()
        return cls._from_compiler(compiler)

    @classmethod
    def _from_compiler(cls, compiler: Compiler) -> tuple[Computation, ...]:
        """Build one view per output of an already-compiled program.

        A caller that also wants the compiler's structural metadata compiles
        once and hands that compiler here, so one trace is never compiled
        twice.
        """
        fusion = _fusion_plan(compiler)
        computations = []
        for index in range(len(compiler.outputs)):
            computation = cls.__new__(cls)
            computation._adopt_program(compiler, index, fusion)
            computations.append(computation)
        return tuple(computations)

    def _adopt_program(
        self,
        compiler: Compiler,
        index: int,
        fusion: tuple[dict[int, Fusion], dict[int, int]],
    ) -> None:
        """Take one output's view of an already-compiled program.

        Every view of one compilation stores the same program objects, so
        multiple outputs share their instructions, slots and fusion metadata
        while each keeps its own view of them.
        """
        self.output = compiler.outputs[index]
        self._variables = compiler.variables
        self._variable_slots = compiler.variable_slots
        self._leaf_slots = compiler.leaf_slots
        self._instructions = compiler.instructions
        self._output_slot = compiler.output_slots[index]
        self._view_slots = compiler.view_slots[index]
        self._view_instructions = compiler.view_instructions[index]
        # Structural nodes are opaque here: they are carried for the public
        # ``nodes`` property and never interpreted.
        self._view_nodes = compiler.view_nodes[index]
        self._fusions, self._fusion_starts = fusion
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

    def _require_active(self) -> None:
        """Reject work after this object has released its graph references."""
        if self._released:
            raise RuntimeError("Computation has been released")

    @property
    def nodes(self) -> list[Node]:
        """Return the cached dependency-first traversal as an independent list.

        The compiler resolved this traversal for this output; a Computation
        only hands it back.
        """
        self._require_active()
        return list(self._view_nodes)

    def release(self) -> None:
        """Release graph references owned by this Computation.

        The output Variable remains usable if the caller retains it, but this
        Computation object cannot be replayed or differentiated afterwards.
        Calling ``release`` more than once is safe.
        """
        if self._released:
            return
        self.output = None
        self._view_nodes = ()
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

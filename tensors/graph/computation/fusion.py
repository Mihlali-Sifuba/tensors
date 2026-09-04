"""Fused execution of compatible instruction runs.

Fusion is an optimization over an already-planned instruction sequence, not
part of the execution model. A :class:`~tensors.graph.computation.Computation`
owns its instructions and decides when to attempt a fused range; this module
recognizes the ranges a backend can execute together and runs them.

Every entry point here may decline. When it does, the Computation executes
exactly the same instructions ordinarily, so the instruction sequence stays
canonical and no result depends on a fusion succeeding.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ...tensor import Tensor
from .gradients import sum_gradient_values

if TYPE_CHECKING:
    from ...variable import Variable
    from .computation import Instruction


#: One fused elementwise step: kernel name, literal scalar, operand order,
#: and the index of an external operand among the fused sources.
FusedStep = tuple[str, float | None, bool, int | None]


#: Fusion metadata for one contiguous run of instructions: the index the run
#: ends at (inclusive), the fused kernel steps, and the slots holding the
#: run's external sources. Fusion is an optimization over the instruction
#: sequence, so it is recorded beside that sequence rather than wrapped
#: around it.
Fusion = tuple[int, tuple[FusedStep, ...], tuple[int, ...]]


#: The operations a fused elementwise kernel implements, by class name.
_KERNEL_NAMES = {
    "Add": "add",
    "Sub": "subtract",
    "Mul": "multiply",
    "Div": "divide",
    "Pow": "power",
    "Neg": "negate",
    "Abs": "abs",
    "Sqrt": "sqrt",
    "Exp": "exp",
    "Log": "log",
    "Sin": "sin",
    "Cos": "cos",
    "Tan": "tan",
    "ArcSin": "arcsin",
    "ArcCos": "arccos",
    "ArcTan": "arctan",
    "Sinh": "sinh",
    "Cosh": "cosh",
    "ArcSinh": "arcsinh",
    "ArcCosh": "arccosh",
    "ArcTanh": "arctanh",
    "Sign": "sign",
    "ReLU": "relu",
    "Sigmoid": "sigmoid",
    "Tanh": "tanh",
    "Softplus": "softplus",
}

#: The fusible kernels that take a second operand.
_BINARY_KERNELS = frozenset({"add", "subtract", "multiply", "divide", "power"})


def fused_operation(instruction: Instruction) -> str | None:
    """Return the internal kernel name for a fusible invocation."""
    # Forward fusion depends only on the forward mathematics, the dtype, and
    # the backend kernel's capability. It never depends on which derivatives
    # a later reverse pass might request.
    return _KERNEL_NAMES.get(type(instruction.operation).__name__)


def start_fusion(
    instruction: Instruction,
    variables: tuple[Variable, ...],
) -> tuple[FusedStep, list[int]] | None:
    """Describe the first operation and source slots of a fused chain."""
    operation = fused_operation(instruction)
    output = variables[instruction.output_slot]
    if operation is None or output.dtype.kind != "floating":
        return None
    if operation not in _BINARY_KERNELS:
        if (
            len(instruction.input_slots) != 1
            or variables[instruction.input_slots[0]].shape != output.shape
        ):
            return None
        return (
            (operation, None, False, None),
            [instruction.input_slots[0]],
        )

    if len(instruction.input_slots) != 2:
        return None
    left_slot, right_slot = instruction.input_slots
    source_slots = [left_slot]
    if right_slot == left_slot:
        operand_index = -1
    else:
        source_slots.append(right_slot)
        operand_index = 1
    return (operation, None, False, operand_index), source_slots


def extend_fusion(
    instruction: Instruction,
    variables: tuple[Variable, ...],
    current_slot: int,
    source_slots: list[int],
    chain_output_slots: set[int],
) -> FusedStep | None:
    """Describe a chain continuation and register external source slots."""
    operation = fused_operation(instruction)
    output = variables[instruction.output_slot]
    if operation is None or output.dtype.kind != "floating":
        return None
    if operation not in _BINARY_KERNELS:
        if instruction.input_slots != (current_slot,):
            return None
        return operation, None, False, None

    if len(instruction.input_slots) != 2:
        return None
    left_slot, right_slot = instruction.input_slots
    if left_slot == current_slot and right_slot == current_slot:
        return operation, None, False, -1
    if left_slot == current_slot:
        external_slot = right_slot
        reverse = False
    elif right_slot == current_slot:
        external_slot = left_slot
        reverse = True
    else:
        return None
    if external_slot in chain_output_slots:
        return None
    try:
        operand_index = source_slots.index(external_slot)
    except ValueError:
        source_slots.append(external_slot)
        operand_index = len(source_slots) - 1
    return operation, None, reverse, operand_index


def plan_fusions(
    instructions: tuple[Instruction, ...],
    variables: tuple[Variable, ...],
) -> tuple[dict[int, Fusion], dict[int, int]]:
    """Find the instruction runs the fused backend can execute together.

    The first result maps the first index of each worthwhile run to its
    metadata, and the second maps the index a run ends at back to the index
    it starts at, because reverse execution recognizes a run by its end. An
    instruction absent from these mappings simply executes ordinarily, so the
    instruction sequence stays canonical and identical across every backend.
    """
    consumer_counts = [0] * len(variables)
    for instruction in instructions:
        for slot in instruction.input_slots:
            consumer_counts[slot] += 1

    fusions: dict[int, Fusion] = {}
    count = len(instructions)
    index = 0
    while index < count:
        opening = start_fusion(instructions[index], variables)
        if opening is None:
            index += 1
            continue

        first_step, source_slots = opening
        first_output = variables[instructions[index].output_slot]
        steps = [first_step]
        current_slot = instructions[index].output_slot
        chain_output_slots = {current_slot}
        end = index
        candidate_index = index + 1
        while candidate_index < count:
            candidate = instructions[candidate_index]
            candidate_output = variables[candidate.output_slot]
            if (
                consumer_counts[current_slot] != 1
                or candidate_output.shape != first_output.shape
                or candidate_output.dtype != first_output.dtype
            ):
                break
            step = extend_fusion(
                candidate,
                variables,
                current_slot,
                source_slots,
                chain_output_slots,
            )
            if step is None:
                break
            steps.append(step)
            current_slot = candidate.output_slot
            chain_output_slots.add(current_slot)
            end = candidate_index
            candidate_index += 1

        if end > index:
            fusions[index] = (end, tuple(steps), tuple(source_slots))
            index = end + 1
        else:
            index += 1

    fusion_starts = {end: start for start, (end, _, _) in fusions.items()}
    return fusions, fusion_starts


def execute_fused_forward(
    start: int,
    fusion: Fusion,
    instructions: tuple[Instruction, ...],
    variables: tuple[Variable, ...],
    values: list[Tensor | None],
) -> bool:
    """Execute one fused instruction run in a single CUDA kernel.

    Returns ``True`` once the fused backend has filled every output slot of
    the run, and ``False`` when the caller must execute the same instructions
    ordinarily instead.
    """
    from ...backend import execute_fused_elementwise

    end, steps, source_slots = fusion
    fused = instructions[start:end + 1]
    first_output = variables[fused[0].output_slot]
    source_values = []
    for slot in source_slots:
        value = values[slot]
        if value is None:
            return False
        source_values.append(value)
    output_shape = first_output.shape
    dtype = first_output.dtype
    storages = execute_fused_elementwise(
        tuple(source_values),
        steps,
        dtype=dtype,
        output_shape=output_shape,
    )
    if storages is None:
        return False
    if len(storages) != len(fused):
        raise RuntimeError(
            "Fused elementwise kernel returned an unexpected output count"
        )
    for instruction, storage in zip(fused, storages):
        output = variables[instruction.output_slot]
        tensor = Tensor._from_owned_storage(
            storage,
            dtype=output.dtype,
            shape=output_shape,
        )
        output._replace_data_from_replay(tensor)
        output._capture_forward_record(
            variables[slot] for slot in instruction.input_slots
        )
        values[instruction.output_slot] = tensor
    return True


def execute_fused_backward(
    start: int,
    fusion: Fusion,
    instructions: tuple[Instruction, ...],
    variables: tuple[Variable, ...],
    gradient_terms: list[list[Any]],
    gradients: list[Any | None],
    live: set[int],
) -> bool:
    """Run a fused instruction run's VJP in a single CUDA kernel.

    Returns ``True`` once the fused backend has served this reverse pass's
    demand for the run, and ``False`` when the caller must differentiate the
    same instructions ordinarily instead.
    """
    from ...backend import execute_fused_elementwise_backward

    end, steps, source_slots = fusion
    fused = instructions[start:end + 1]
    last_output = variables[fused[-1].output_slot]
    output_terms = gradient_terms[fused[-1].output_slot]
    if not output_terms:
        return True
    # A fused range accumulates gradient terms exactly as the ordinary
    # reverse pass does.
    output_gradient = sum_gradient_values(output_terms)
    source_values = tuple(variables[slot].data for slot in source_slots)
    output_shape = last_output.shape
    dtype = last_output.dtype
    # An external operand gradient is only calculated where this reverse
    # pass wants it. The chain's internal derivative is separate: reverse
    # propagation always needs it to reach the chain's first input.
    external_steps = tuple(
        (step_index, operand_index)
        for step_index, (_, scalar, _, operand_index) in enumerate(steps)
        if scalar is None
        and operand_index is not None
        and operand_index >= 0
        and source_slots[operand_index] in live
    )
    storages = execute_fused_elementwise_backward(
        source_values,
        output_gradient,
        steps,
        dtype=dtype,
        output_shape=output_shape,
        requested_external=tuple(
            step_index for step_index, _ in external_steps
        ),
    )
    if storages is None:
        # The fused VJP cannot supply a requested derivative, so ordinary
        # operation execution handles this group instead.
        return False

    expected = len(fused) + 1 + len(external_steps)
    if len(storages) != expected:
        raise RuntimeError(
            "Fused elementwise VJP returned an unexpected output count"
        )

    for instruction, storage in zip(fused, storages):
        gradients[instruction.output_slot] = Tensor._from_owned_storage(
            storage,
            dtype=dtype,
            shape=output_shape,
        )

    def append_source_gradient(storage: Any, source_index: int) -> None:
        slot = source_slots[source_index]
        if slot not in live:
            return
        variable = variables[slot]
        gradient = Tensor._from_owned_storage(storage, dtype=dtype, shape=output_shape)
        if gradient.shape != variable.shape:
            from ...ops._utils import sum_to_shape

            gradient = sum_to_shape(gradient, variable.shape)
        if gradient.dtype != variable.dtype:
            gradient = gradient.astype(variable.dtype)
        gradient_terms[slot].append(gradient)

    append_source_gradient(storages[len(fused)], 0)
    external_offset = len(fused) + 1
    for row, (_, operand_index) in enumerate(external_steps):
        append_source_gradient(
            storages[external_offset + row],
            operand_index,
        )
    return True

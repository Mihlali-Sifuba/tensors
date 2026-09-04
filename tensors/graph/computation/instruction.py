"""One executable operation invocation."""

from __future__ import annotations

from dataclasses import dataclass

from ...ops.operation import Operation


@dataclass(frozen=True, slots=True)
class Instruction:
    """One executable operation invocation.

    An instruction says which :class:`Operation` runs, which slots hold its
    operands, and which slot receives its result. It is what a
    :class:`~tensors.graph.computation.compiler.Compiler` emits and what a
    :class:`~tensors.graph.computation.computation.Computation` executes.
    Everything else about the invocation — the Variables themselves, their
    mutation records, and which of its VJPs a reverse pass wants — is
    resolved by the Computation that owns the instruction.
    """

    operation: Operation
    input_slots: tuple[int, ...]
    output_slot: int

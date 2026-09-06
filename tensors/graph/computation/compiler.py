"""Translation of a recorded graph into an executable instruction program.

A recorded graph is structure: Variable and operation vertices joined by
edges. Execution wants something flatter — a numbered slot per value and an
ordered sequence of :class:`Instruction` objects over those slots. The
:class:`Compiler` performs exactly that translation and nothing else.

It is the boundary between the two domains: it is the last component that
understands Nodes and Edges, and it does not know what will run the program
it emits. Compilation is structural throughout, because a slot belongs to a
:class:`~tensors.graph.node.VariableNode` rather than to the runtime Variable
that vertex may not have been materialized as yet. Structural metadata for
the graph layer and execution metadata for the runtime both come out of one
compilation, so neither side has to reconstruct the other's view afterwards.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

from ..node import Node, VariableNode
from .instruction import Instruction

if TYPE_CHECKING:
    from ...variable import Variable
    from ..edge import Edge


def resolve_outputs(outputs: Iterable[Variable]) -> tuple[VariableNode, ...]:
    """Return the vertices a set of runtime outputs is compiled from.

    This is where the runtime hands its values to the graph: a caller names
    outputs as Variables, and everything from here on works in terms of the
    vertices naming them.
    """
    nodes = tuple(_graph_identity(output, "output") for output in outputs)
    if not nodes:
        raise ValueError("Computation requires at least one output")
    return nodes


def resolve_boundaries(
    boundaries: Iterable[Variable],
) -> tuple[VariableNode, ...]:
    """Return the vertices whose graphs a traversal must stop at."""
    return tuple(
        _graph_identity(boundary, "boundary") for boundary in boundaries
    )


def _graph_identity(value: Variable, role: str) -> VariableNode:
    """Return the vertex naming ``value``, rejecting an ungraphed one."""
    node = getattr(value, "node", None)
    if not isinstance(node, VariableNode):
        raise TypeError(f"Computation {role} must have a graph node")
    return node


def _validated_outputs(
    outputs: tuple[VariableNode, ...],
) -> tuple[VariableNode, ...]:
    """Reject an output set no graph can be compiled from."""
    if not outputs:
        raise ValueError("Compiler requires at least one output")
    for output in outputs:
        if not isinstance(output, VariableNode):
            raise TypeError(
                "Compiler output must be a VariableNode, got "
                f"{type(output).__name__}"
            )
    return outputs


class Compiler:
    """Compiles the graph reaching a set of output vertices into instructions.

    One compilation serves every requested output: the traversal, the slot
    numbering, and the instruction sequence are shared, and per-output
    reachability is recorded as a bit mask so a caller can tell which part of
    the program each output needs. Boundary vertices end the traversal, so the
    graph behind one compiles into a leaf rather than into instructions.

    A slot is the compiled identity of a value, and it is numbered by the
    vertex naming that value. A graph therefore compiles whether or not its
    values exist: a result no execution has produced yet still receives a
    slot, still ends an instruction, and still resolves into each view.

    :meth:`compile` returns the instruction sequence. The metadata that
    sequence is expressed in terms of stays readable on the compiler: the
    slots and leaves the runtime executes over, the per-output execution
    views resolved from the reachability masks, and the traversal and edges
    the graph layer keeps as its own structural record. :attr:`variables` and
    :attr:`variable_slots` project the finished program onto the runtime
    Variables occupying its slots, and are the only part of a compilation
    that requires those values to have been materialized.
    """

    def __init__(
        self,
        outputs: Iterable[VariableNode],
        *,
        boundaries: Iterable[VariableNode] = (),
    ) -> None:
        #: The vertices compilation was requested for, in request order.
        self.output_nodes = _validated_outputs(tuple(outputs))
        #: The vertices the traversal stops at.
        self.boundary_nodes: frozenset[Node] = frozenset(boundaries)
        #: The dependency-first traversal reaching every output.
        self.nodes: tuple[Node, ...] = ()
        #: One bit per output, set where that output reaches the node.
        self.node_masks: tuple[int, ...] = ()
        #: Every value vertex the program names, in slot order.
        self.variable_nodes: tuple[VariableNode, ...] = ()
        #: The slot each value vertex occupies.
        self.node_slots: dict[VariableNode, int] = {}
        #: The slots holding values the program reads but never produces.
        self.leaf_slots: tuple[int, ...] = ()
        #: The slot each requested output is produced into.
        self.output_slots: tuple[int, ...] = ()
        #: The compiled program, in dependency order.
        self.instructions: tuple[Instruction, ...] = ()
        #: Each output's reachable nodes, as structural metadata to pass on.
        self.view_nodes: tuple[tuple[Node, ...], ...] = ()
        #: The slots each output's execution reaches, in slot order.
        self.view_slots: tuple[tuple[int, ...], ...] = ()
        #: The instructions each output's execution reaches, in program order.
        self.view_instructions: tuple[tuple[Instruction, ...], ...] = ()
        self._edges: tuple[Edge, ...] | None = None
        self._variables: tuple[Variable, ...] | None = None
        self._variable_slots: dict[Variable, int] | None = None

    def compile(self) -> tuple[Instruction, ...]:
        """Return the instruction sequence the recorded graph compiles to.

        Every value vertex becomes an execution slot. Every operation vertex
        becomes one instruction whose operands are named by its incoming
        edges and whose result is the vertex named by its outgoing edge.
        Execution then works from this compact program instead of walking the
        graph again.
        """
        self._resolve_dependencies()
        self._assign_slots()
        self._emit_instructions()
        self._resolve_views()
        return self.instructions

    @property
    def edges(self) -> tuple[Edge, ...]:
        """Return the recorded data flow between the traversed nodes.

        Only the graph layer keeps this, so it is calculated on request
        rather than charged to every compilation.
        """
        if self._edges is None:
            boundary_nodes = self.boundary_nodes
            # The traversal already excludes anything past a boundary, so the
            # incoming edges of the traversed vertices are exactly the
            # recorded data flow.
            self._edges = tuple(
                edge
                for node in self.nodes
                if node not in boundary_nodes
                for edge in node._in_edges
            )
        return self._edges

    def _resolve_dependencies(self) -> None:
        """Traverse the graph once and record per-output reachability."""
        self.nodes = self._traverse()
        self.node_masks = self._reachability_masks()

    def _traverse(self) -> tuple[Node, ...]:
        """Return the dependency-first traversal reaching every output."""
        boundary_nodes = self.boundary_nodes
        order: list[Node] = []
        visited: set[Node] = set()
        for output in self.output_nodes:
            stack: list[tuple[Node, bool]] = [(output, False)]
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
        return tuple(order)

    def _reachability_masks(self) -> tuple[int, ...]:
        """Return which outputs reach each traversed node, one bit each."""
        boundary_nodes = self.boundary_nodes
        masks = {node: 0 for node in self.nodes}
        for index, output in enumerate(self.output_nodes):
            masks[output] |= 1 << index
        for node in reversed(self.nodes):
            mask = masks[node]
            if not mask or node in boundary_nodes:
                continue
            for edge in node._in_edges:
                masks[edge.source] |= mask
        return tuple(masks[node] for node in self.nodes)

    def _assign_slots(self) -> None:
        """Number every value vertex the traversal reached.

        A slot is numbered by vertex identity, so nothing here reads a value.
        """
        variable_nodes = tuple(
            node for node in self.nodes if isinstance(node, VariableNode)
        )
        self.variable_nodes = variable_nodes
        self.node_slots = {
            node: index for index, node in enumerate(variable_nodes)
        }
        self.output_slots = tuple(
            self.node_slots[output] for output in self.output_nodes
        )

    def _emit_instructions(self) -> None:
        """Emit one instruction per produced value, in dependency order."""
        slots = self.node_slots
        boundary_nodes = self.boundary_nodes
        instructions: list[Instruction] = []
        leaf_slots: list[int] = []
        for node in self.variable_nodes:
            output_slot = slots[node]
            producer = node.producer
            if producer is None or node in boundary_nodes:
                leaf_slots.append(output_slot)
                continue
            instructions.append(
                Instruction(
                    operation=producer.operation,
                    input_slots=tuple(
                        slots[operand] for operand in producer.operand_nodes
                    ),
                    output_slot=output_slot,
                )
            )
        self.leaf_slots = tuple(leaf_slots)
        self.instructions = tuple(instructions)

    def _resolve_views(self) -> None:
        """Translate each output's reachability into its execution view.

        Reachability is a fact about graph structure; a view is a fact about
        the compiled program. Resolving one into the other here is what lets
        the runtime hold an execution view without reading the graph.
        """
        slots = self.node_slots
        view_nodes: list[tuple[Node, ...]] = []
        view_slots: list[tuple[int, ...]] = []
        view_instructions: list[tuple[Instruction, ...]] = []
        for index in range(len(self.output_nodes)):
            output_bit = 1 << index
            nodes = tuple(
                node
                for node, mask in zip(self.nodes, self.node_masks)
                if mask & output_bit
            )
            reached = {
                slots[node]
                for node in nodes
                if isinstance(node, VariableNode)
            }
            view_nodes.append(nodes)
            view_slots.append(tuple(sorted(reached)))
            view_instructions.append(
                tuple(
                    instruction
                    for instruction in self.instructions
                    if instruction.output_slot in reached
                )
            )
        self.view_nodes = tuple(view_nodes)
        self.view_slots = tuple(view_slots)
        self.view_instructions = tuple(view_instructions)

    @property
    def variables(self) -> tuple[Variable, ...]:
        """Return the runtime Variable occupying each slot, in slot order.

        This is the projection of a compiled program back onto the runtime,
        and the one part of a compilation that requires its values to exist.
        It is resolved on request so that compiling never depends on it.
        """
        variables = self._variables
        if variables is None:
            variables = tuple(node.variable for node in self.variable_nodes)
            self._variables = variables
        return variables

    @property
    def variable_slots(self) -> dict[Variable, int]:
        """Return the slot each runtime Variable occupies.

        The runtime looks slots up by the value it holds; compilation numbers
        them by vertex. This resolves the first from the second, and like
        :attr:`variables` requires every slot to have been materialized.
        """
        slots = self._variable_slots
        if slots is None:
            slots = {
                variable: index
                for index, variable in enumerate(self.variables)
            }
            self._variable_slots = slots
        return slots

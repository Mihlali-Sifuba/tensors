"""Translation of a recorded graph into an executable instruction program.

A recorded graph is structure: Variable and operation vertices joined by
edges. Execution wants something flatter — a numbered slot per Variable and
an ordered sequence of :class:`Instruction` objects over those slots. The
:class:`Compiler` performs exactly that translation and nothing else.

It is the boundary between the two domains: it is the last component that
understands Nodes and Edges, and it does not know what will run the program
it emits. Structural metadata for the graph layer and execution metadata for
the runtime both come out of one compilation, so neither side has to
reconstruct the other's view afterwards.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

from ..node import Node, VariableNode
from .instruction import Instruction

if TYPE_CHECKING:
    from ...variable import Variable
    from ..edge import Edge


def validate_outputs(outputs: tuple[Variable, ...]) -> tuple[Variable, ...]:
    """Reject outputs a recorded graph cannot be compiled from."""
    if not outputs:
        raise ValueError("Computation requires at least one output")
    for output in outputs:
        if not isinstance(getattr(output, "node", None), VariableNode):
            raise TypeError("Computation output must have a graph node")
    return outputs


class Compiler:
    """Compiles the graph reaching a set of outputs into instructions.

    One compilation serves every requested output: the traversal, the slot
    numbering, and the instruction sequence are shared, and per-output
    reachability is recorded as a bit mask so a caller can tell which part of
    the program each output needs. Boundary Variables end the traversal, so
    the graph behind one compiles into a leaf rather than into instructions.

    :meth:`compile` returns the instruction sequence. The metadata that
    sequence is expressed in terms of stays readable on the compiler: the
    slots and leaves the runtime executes over, the per-output execution
    views resolved from the reachability masks, and the traversal and edges
    the graph layer keeps as its own structural record.
    """

    def __init__(
        self,
        outputs: Iterable[Variable],
        *,
        boundaries: Iterable[Variable] = (),
    ) -> None:
        self.outputs = validate_outputs(tuple(outputs))
        self.boundaries = tuple(boundaries)
        #: The dependency-first traversal reaching every output.
        self.nodes: tuple[Node, ...] = ()
        #: One bit per output, set where that output reaches the node.
        self.node_masks: tuple[int, ...] = ()
        #: The nodes the traversal stopped at.
        self.boundary_nodes: frozenset[Node] = frozenset()
        #: Every Variable the program names, in slot order.
        self.variables: tuple[Variable, ...] = ()
        #: The slot each Variable occupies.
        self.variable_slots: dict[Variable, int] = {}
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

    def compile(self) -> tuple[Instruction, ...]:
        """Return the instruction sequence the recorded graph compiles to.

        Every Variable vertex becomes an execution slot. Every operation
        vertex becomes one instruction whose operands are named by its
        incoming edges and whose result is the Variable named by its outgoing
        edge. Execution then works from this compact program instead of
        walking the graph again.
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
        self.boundary_nodes = frozenset(
            variable.node
            for variable in self.boundaries
            if getattr(variable, "node", None) is not None
        )
        self.nodes = self._traverse()
        self.node_masks = self._reachability_masks()

    def _traverse(self) -> tuple[Node, ...]:
        """Return the dependency-first traversal reaching every output."""
        boundary_nodes = self.boundary_nodes
        order: list[Node] = []
        visited: set[Node] = set()
        for output in self.outputs:
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
        return tuple(order)

    def _reachability_masks(self) -> tuple[int, ...]:
        """Return which outputs reach each traversed node, one bit each."""
        boundary_nodes = self.boundary_nodes
        masks = {node: 0 for node in self.nodes}
        for index, output in enumerate(self.outputs):
            masks[output.node] |= 1 << index
        for node in reversed(self.nodes):
            mask = masks[node]
            if not mask or node in boundary_nodes:
                continue
            for edge in node._in_edges:
                masks[edge.source] |= mask
        return tuple(masks[node] for node in self.nodes)

    def _assign_slots(self) -> None:
        """Number every Variable the traversal reached."""
        variables = tuple(
            node.variable
            for node in self.nodes
            if isinstance(node, VariableNode)
        )
        self.variables = variables
        self.variable_slots = {
            variable: index for index, variable in enumerate(variables)
        }
        self.output_slots = tuple(
            self.variable_slots[output] for output in self.outputs
        )

    def _emit_instructions(self) -> None:
        """Emit one instruction per produced Variable, in dependency order."""
        slots = self.variable_slots
        boundary_nodes = self.boundary_nodes
        instructions: list[Instruction] = []
        leaf_slots: list[int] = []
        for node in self.nodes:
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
        self.leaf_slots = tuple(leaf_slots)
        self.instructions = tuple(instructions)

    def _resolve_views(self) -> None:
        """Translate each output's reachability into its execution view.

        Reachability is a fact about graph structure; a view is a fact about
        the compiled program. Resolving one into the other here is what lets
        the runtime hold an execution view without reading the graph.
        """
        slots = self.variable_slots
        view_nodes: list[tuple[Node, ...]] = []
        view_slots: list[tuple[int, ...]] = []
        view_instructions: list[tuple[Instruction, ...]] = []
        for index in range(len(self.outputs)):
            output_bit = 1 << index
            nodes = tuple(
                node
                for node, mask in zip(self.nodes, self.node_masks)
                if mask & output_bit
            )
            reached = {
                slots[node.variable]
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

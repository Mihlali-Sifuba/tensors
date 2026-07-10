"""Node — a vertex in the computation graph."""


class Node:
    """A vertex in the computation graph.

    Each node represents either:
    - A **leaf variable** (label ``'var'``) — user-provided input.
    - An **operation** (label like ``'add'``, ``'mul'``) — transforms inputs
      into an output.

    Nodes are connected by :class:`~tensors.autograd.edge.Edge` objects.
    The *output_var* attribute links back to the
    :class:`~tensors.autograd.variable.Variable` that this node produces.
    """

    _next_id = 0

    def __init__(self, label=None, output_var=None, **kwargs):
        self.id = Node._next_id
        Node._next_id += 1

        self.label = label               # e.g. "var", "add", "mul", "dot"
        self.output_var = output_var     # the Variable this node produces
        self.args = kwargs               # extra metadata for backward

        self._in_edges = []   # edges from input nodes → this node
        self._out_edges = []  # edges from this node → output nodes

    # -- convenience accessors -----------------------------------------

    @property
    def inputs(self):
        """Input nodes (predecessors)."""
        return [e.source for e in self._in_edges]

    @property
    def outputs(self):
        """Output nodes (successors)."""
        return [e.target for e in self._out_edges]

    # -- dunder methods ------------------------------------------------

    def __repr__(self):
        lbl = self.label or "?"
        return f"Node({lbl}, #{self.id})"

    def __hash__(self):
        return self.id

    def __eq__(self, other):
        return isinstance(other, Node) and self.id == other.id

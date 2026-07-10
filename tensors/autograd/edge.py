"""Edge — a directed connection between two Nodes in the computation graph."""


class Edge:
    """A directed connection from one Node to another.

    Each edge represents a data flow: the *source* node produces a value
    that is consumed by the *target* node. The optional *label* describes
    the role (e.g. ``'a'``, ``'b'`` for binary ops, or ``'input_0'``).

    Edges are automatically registered with both source and target nodes
    upon creation.
    """

    def __init__(self, source, target, label=None):
        """
        Args:
            source: The upstream node (produces a value).
            target: The downstream node (consumes the value).
            label: Optional descriptor (e.g. ``'a'``, ``'b'``, ``'input'``).
        """
        self.source = source
        self.target = target
        self.label = label

        source._out_edges.append(self)
        target._in_edges.append(self)

    def __repr__(self):
        src = self.source.label or "?"
        tgt = self.target.label or "?"
        lbl = f" '{self.label}'" if self.label else ""
        return f"Edge({src} → {tgt}{lbl})"

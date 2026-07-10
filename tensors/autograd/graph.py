"""Implicit computation graph with explicit execution context."""

import threading
from array import array

from ..tensor import Tensor
from .node import Node
from .edge import Edge


# ---------------------------------------------------------------------------
#  Thread-local implicit graph
# ---------------------------------------------------------------------------

_local = threading.local()


def _get_graph():
    """Get the current thread's implicit graph (creates it if needed)."""
    if not hasattr(_local, "graph"):
        _local.graph = _GraphState()
    return _local.graph


def _reset_graph():
    """Clear the current thread's graph."""
    _local.graph = _GraphState()


# ---------------------------------------------------------------------------
#  Graph state (internal) — accumulates nodes and edges
# ---------------------------------------------------------------------------

class _GraphState:
    """Thread-local graph state — accumulates nodes and edges implicitly."""

    __slots__ = ("nodes", "edges")

    def __init__(self):
        self.nodes = []       # list of Node
        self.edges = []       # list of Edge

    def add_node(self, label, output_var=None, op_cls=None, **kwargs):
        """Create and register a new Node."""
        node = Node(label=label, output_var=output_var, op_cls=op_cls, **kwargs)
        self.nodes.append(node)
        return node

    def add_edge(self, source, target, label=None):
        """Create and register a new Edge between two nodes."""
        if source not in self.nodes:
            self.nodes.append(source)
        if target not in self.nodes:
            self.nodes.append(target)
        edge = Edge(source, target, label=label)
        self.edges.append(edge)
        return edge


# ---------------------------------------------------------------------------
#  Graph — explicit execution context
# ---------------------------------------------------------------------------

class Graph:
    """Execution context for forward/backward passes.

    Usage::

        x = Variable([1.0, 2.0])
        w = Variable([0.5, 0.5])
        y = x * w + 1.0

        with Graph() as g:
            g.backward(y)

        print(w.grad)   # Tensor([1.0, 2.0])
    """

    def __init__(self):
        self._state = _get_graph()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        # Capture snapshot for inspection before reset
        self._captured_nodes = list(self._state.nodes)
        self._captured_edges = list(self._state.edges)
        _reset_graph()

    @property
    def nodes(self):
        """Inspect the computation nodes captured during this execution."""
        if not hasattr(self, "_captured_nodes"):
            return list(self._state.nodes)
        return list(self._captured_nodes)

    @property
    def edges(self):
        """Inspect the computation edges captured during this execution."""
        if not hasattr(self, "_captured_edges"):
            return list(self._state.edges)
        return list(self._captured_edges)

    # ------------------------------------------------------------------
    #  Forward
    # ------------------------------------------------------------------

    def forward(self, output_var=None):
        """Evaluate the graph in topological order.

        Args:
            output_var: If given, returns only this Variable's value.

        Returns:
            A dict of {Variable: Tensor} or a single Tensor.
        """
        output_node = output_var.node if output_var is not None else None
        order = self._topological_sort(output_node)
        values = {}
        for node in order:
            if node.label == "var":
                values[node.output_var] = node.output_var.data
            else:
                result = self._exec(node, values)
                node.output_var.data = result
                values[node.output_var] = result
        return values.get(output_var) if output_var is not None else values

    # ------------------------------------------------------------------
    #  Backward
    # ------------------------------------------------------------------

    def backward(self, loss_var, grad=None):
        """Compute gradients via reverse-mode autodiff.

        Args:
            loss_var: The Variable to compute gradients from.
        """
        order = self._topological_sort(loss_var.node)

        # Each call is independent. This prevents stale intermediate gradients
        # from amplifying subsequent backward passes.
        for node in order:
            if node.output_var is not None:
                node.output_var.grad = None

        # Seed gradient — ones matching loss shape unless supplied explicitly.
        loss_shape = loss_var.data.shape
        if grad is None:
            typecode = (
                loss_var.dtype.typecode
                if loss_var.dtype.typecode in {"f", "d"}
                else "d"
            )
            seed_data = array(typecode, [1.0] * loss_var.data.size)
            loss_var.grad = Tensor(seed_data, shape=loss_shape)
        else:
            seed = grad if isinstance(grad, Tensor) else Tensor(grad)
            if seed.shape != loss_shape:
                raise ValueError(
                    f"Gradient shape {seed.shape} does not match loss shape {loss_shape}"
                )
            loss_var.grad = seed

        # Walk nodes in reverse topological order
        for node in reversed(order):
            if node.label == "var" or node.op_cls is None:
                continue

            out_var = node.output_var
            out_grad = out_var.grad
            if out_grad is None:
                continue

            # Get input data tensors from incoming edges
            in_data = [e.source.output_var.data for e in node._in_edges]

            # Generic dispatch: op_cls knows its own backward
            grads = node.op_cls.backward(out_grad, *in_data, **node.args)
            for v, g in zip(
                [e.source.output_var for e in node._in_edges], grads
            ):
                self._acc_grad(v, g)

    # ------------------------------------------------------------------
    #  Helpers
    # ------------------------------------------------------------------

    def _topological_sort(self, output_node=None):
        """Return nodes in topological order (dependencies first)."""
        order = []
        visited = set()

        def dfs(node):
            if node in visited:
                return
            visited.add(node)
            for in_edge in node._in_edges:
                dfs(in_edge.source)
            order.append(node)

        roots = [output_node] if output_node is not None else self._state.nodes
        for node in roots:
            dfs(node)
        return order

    @staticmethod
    def _acc_grad(var, grad):
        """Accumulate gradient into a Variable."""
        if not var.requires_grad:
            return
        if var.grad is None:
            var.grad = grad
        else:
            from ..ops import Ops
            var.grad = Ops.add(var.grad, grad)

    def _exec(self, node, values):
        """Execute a single node during forward using op_cls.forward."""
        in_vars = [e.source.output_var for e in node._in_edges]
        args = [values[v] for v in in_vars]
        if "scalar" in node.args:
            scalar = node.args["scalar"]
            if node.args.get("reverse", False):
                result = node.op_cls.forward_reverse(args[0], scalar)
            else:
                result = node.op_cls.forward(args[0], scalar)
        elif "key" in node.args:
            result = node.op_cls.forward(args[0], node.args["key"])
        else:
            result = node.op_cls.forward(*args)
        if not isinstance(result, Tensor):
            result = Tensor([result])
        return result

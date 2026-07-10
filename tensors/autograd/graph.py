"""Implicit computation graph with explicit execution context."""

import threading
from array import array

from ..tensor import Tensor
from .. import dtype as _dtype
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
    if hasattr(_local, "graph"):
        _local.graph.nodes.clear()
        _local.graph.edges.clear()


# ---------------------------------------------------------------------------
#  Graph state (internal) — accumulates nodes and edges
# ---------------------------------------------------------------------------

class _GraphState:
    """Thread-local graph state — accumulates nodes and edges implicitly."""

    __slots__ = ("nodes", "edges")

    def __init__(self):
        self.nodes = []       # list of Node
        self.edges = []       # list of Edge

    def add_node(self, label, output_var=None, **kwargs):
        """Create and register a new Node."""
        node = Node(label=label, output_var=output_var, **kwargs)
        self.nodes.append(node)
        return node

    def add_edge(self, source, target, label=None):
        """Create and register a new Edge between two nodes."""
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

        print(w.grad)   # Tensor([0.5, 0.5])
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
            return []
        return list(self._captured_nodes)

    @property
    def edges(self):
        """Inspect the computation edges captured during this execution."""
        if not hasattr(self, "_captured_edges"):
            return []
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
        order = self._topological_sort()
        values = {}
        for node in order:
            if node.label == "var":
                values[node.output_var] = node.output_var.data
            else:
                values[node.output_var] = self._exec(node, values)
        return values.get(output_var) if output_var else values

    # ------------------------------------------------------------------
    #  Backward
    # ------------------------------------------------------------------

    def backward(self, loss_var):
        """Compute gradients via reverse-mode autodiff.

        Args:
            loss_var: The Variable to compute gradients from.
        """
        # Seed gradient — ones matching loss shape
        from array import array
        loss_shape = loss_var.data.shape
        seed_data = array(loss_var.dtype.typecode, [1.0] * loss_var.data.size)
        loss_var.grad = Tensor(seed_data, shape=loss_shape)

        # Walk nodes in reverse topological order
        for node in reversed(self._state.nodes):
            if node.label == "var":
                continue

            out_var = node.output_var
            out_grad = out_var.grad
            if out_grad is None:
                continue

            # Get input variables from incoming edges
            in_vars = [e.source.output_var for e in node._in_edges]

            # Dispatch backward
            if node.label == "add":
                for v in in_vars:
                    self._acc_grad(v, out_grad)

            elif node.label == "sub":
                for i, v in enumerate(in_vars):
                    if i == 0:
                        self._acc_grad(v, out_grad)
                    else:
                        self._acc_grad(v, _neg_tensor(out_grad))

            elif node.label == "mul":
                if len(in_vars) > 1:
                    a, b = in_vars
                    self._acc_grad(a, _elementwise_mul(out_grad, b.data))
                    self._acc_grad(b, _elementwise_mul(out_grad, a.data))
                else:
                    scalar_val = node.args.get("scalar", 1.0)
                    self._acc_grad(in_vars[0], _scalar_mul(out_grad, scalar_val))

            elif node.label == "div":
                if len(in_vars) > 1:
                    a, b = in_vars
                    self._acc_grad(a, _elementwise_div(out_grad, b.data))
                    self._acc_grad(b, _neg_tensor(_elementwise_div(
                        _elementwise_mul(out_grad, a.data),
                        _elementwise_mul(b.data, b.data)
                    )))
                else:
                    scalar_val = node.args.get("scalar", 1.0)
                    self._acc_grad(in_vars[0], _scalar_mul(out_grad, 1.0 / scalar_val))

            elif node.label == "neg":
                self._acc_grad(in_vars[0], _neg_tensor(out_grad))

            elif node.label == "sum":
                a = in_vars[0]
                self._acc_grad(a, Tensor(
                    [1.0] * a.data.size, dtype=a.dtype.typecode,
                    shape=a.data.shape
                ))

            elif node.label == "mean":
                a = in_vars[0]
                scale = 1.0 / a.data.size
                in_shape = a.data.shape
                g = _scalar_mul(out_grad, scale)
                if g.shape != in_shape:
                    if g.size == _total_elements(in_shape):
                        g = Tensor(g._data, shape=in_shape)
                self._acc_grad(a, g)

            elif node.label == "dot":
                a, b = in_vars
                from ..ops import Ops as _Ops
                og = out_grad
                if og.ndim == 1:
                    og = Tensor(og._data, shape=(1, og.shape[0]))
                self._acc_grad(a, _Ops.dot(og, _Ops.transpose(b.data)))
                self._acc_grad(b, _Ops.dot(_Ops.transpose(a.data), og))

    # ------------------------------------------------------------------
    #  Helpers
    # ------------------------------------------------------------------

    def _topological_sort(self):
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

        for node in self._state.nodes:
            dfs(node)
        return order

    @staticmethod
    def _acc_grad(var, grad):
        """Accumulate gradient into a Variable."""
        if var.grad is None:
            var.grad = grad
        else:
            from ..ops import Ops
            var.grad = Ops.add(var.grad, grad)

    def _exec(self, node, values):
        """Execute a single node during forward."""
        from ..ops import Ops
        in_vars = [e.source.output_var for e in node._in_edges]

        a_val = values[in_vars[0]]
        b_val = values[in_vars[1]] if len(in_vars) > 1 else None

        if node.label == "add":
            if b_val is not None:
                return Ops.add(a_val, b_val)
            return a_val
        elif node.label == "sub":
            if b_val is not None:
                return Ops.subtract(a_val, b_val)
            return a_val
        elif node.label == "mul":
            if b_val is not None:
                return Ops.multiply(a_val, b_val)
            return a_val
        elif node.label == "div":
            if b_val is not None:
                return Ops.divide(a_val, b_val)
            return a_val
        elif node.label == "neg":
            return Ops.multiply(a_val, -1)
        elif node.label == "dot":
            return Ops.dot(a_val, b_val)
        elif node.label == "sum":
            return Tensor([Ops.sum(a_val)])
        elif node.label == "mean":
            return Tensor([Ops.mean(a_val)])
        else:
            raise RuntimeError(f"Unknown label: {node.label}")


# ---------------------------------------------------------------------------
#  Gradient helper functions (small Tensor ops to avoid circular imports)
# ---------------------------------------------------------------------------

def _neg_tensor(t):
    return Tensor([-x for x in t._data], dtype=t.dtype.typecode, shape=t.shape)


def _scalar_mul(t, s):
    return Tensor([x * s for x in t._data], dtype=t.dtype.typecode, shape=t.shape)


def _elementwise_mul(a, b):
    from ..ops import Ops
    if isinstance(b, (int, float)):
        return _scalar_mul(a, b)
    return Ops.multiply(a, b)


def _elementwise_div(a, b):
    from ..ops import Ops
    return Ops.divide(a, b)


def _total_elements(shape):
    from math import prod
    return prod(shape)

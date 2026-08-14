"""Callable, reusable computational graph functions.

``Graph`` is the opt-in model abstraction.  A subclass defines ``forward``;
passing a function to ``Graph`` creates the equivalent functional model and
allows ``@Graph`` decorator use.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any
from inspect import getclosurevars

from ..variable import Variable
from .computation import Computation
from .state import TraceScope


class Graph:
    """A callable differentiable model that records its latest computation.

    Every call executes ``forward`` eagerly and captures the Variables
    reachable from that call's output.  A subclass implements ``forward``;
    ``Graph(function)`` and ``@Graph`` provide the functional form.
    """

    def __init__(self, function: Callable[..., Any] | None = None) -> None:
        if function is not None and not callable(function):
            raise TypeError("Graph expects a callable function or no argument")

        self._function = function
        self._outputs = None
        self._computations: tuple[Computation, ...] = ()
        self._nodes = ()
        self._edges = ()

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Define a subclass model's eager computation."""
        if self._function is None:
            raise NotImplementedError("Graph subclasses must implement forward()")
        return self._function(*args, **kwargs)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Execute ``forward`` and record a fresh computation."""
        scope = TraceScope()
        try:
            return self._trace(args, kwargs)
        finally:
            scope.close()

    def rebuild(self, *args: Any, **kwargs: Any) -> Any:
        """Execute ``forward`` and record a fresh computation."""
        return self(*args, **kwargs)

    def release(self) -> None:
        """Release this Graph's references to its latest computation.

        Returned output Variables remain valid when retained by the caller.
        The Graph can be called again to record a new computation.
        """
        self._outputs = None
        self._computations = ()
        self._nodes = ()
        self._edges = ()

    @property
    def nodes(self) -> list[Any]:
        """Nodes reachable from this graph's output variables."""
        return list(self._nodes)

    @property
    def edges(self) -> list[Any]:
        """Edges reachable from this graph's output variables."""
        return list(self._edges)

    @property
    def computation(self) -> Computation:
        """Return the single computation produced by this graph."""
        if not self._computations:
            raise RuntimeError("Graph has not been called yet")
        if len(self._computations) != 1:
            raise RuntimeError("Graph has multiple outputs; use computations")
        return self._computations[0]

    @property
    def computations(self) -> tuple[Computation, ...]:
        """Return computations for every flattened graph output."""
        return self._computations

    def parameters(self) -> list[Variable]:
        """Return persistent trainable Variables owned by this graph.

        Child Graph instances are traversed recursively.  Function graphs also
        inspect their closure so ``@Graph`` can capture simple parameters.
        """
        found = []
        visited = set()

        def visit(value: Any) -> None:
            identity = id(value)
            if identity in visited:
                return
            visited.add(identity)

            if isinstance(value, Variable):
                if value.requires_grad:
                    found.append(value)
                return

            if isinstance(value, Graph):
                for name, child in vars(value).items():
                    if not name.startswith("_"):
                        visit(child)
                if value._function is not None:
                    for captured in self._function_values(value._function):
                        visit(captured)
                return

            if isinstance(value, dict):
                for item in value.values():
                    visit(item)
            elif isinstance(value, (list, tuple, set)):
                for item in value:
                    visit(item)

        for name, value in vars(self).items():
            if not name.startswith("_"):
                visit(value)
        if self._function is not None:
            for captured in self._function_values(self._function):
                visit(captured)
        return found

    def _trace(self, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        input_args = tuple(self._as_input(value) for value in args)
        keyword_names = tuple(kwargs)
        input_kwargs = {
            name: self._as_input(kwargs[name]) for name in keyword_names
        }

        outputs = self.forward(*input_args, **input_kwargs)
        output_vars = tuple(self._iter_output_variables(outputs))
        if not output_vars:
            raise TypeError(
                "Graph.forward() must return a Variable or a tuple/list of Variables"
            )

        self._outputs = outputs
        self._computations = tuple(Computation(output) for output in output_vars)
        self._nodes, self._edges = self._capture(self._computations)
        return outputs

    @staticmethod
    def _as_input(value: Any) -> Variable:
        if isinstance(value, Variable):
            return value
        return Variable(value, requires_grad=False)

    @staticmethod
    def _iter_output_variables(outputs: Any) -> Iterator[Variable]:
        if isinstance(outputs, Variable):
            yield outputs
            return
        if isinstance(outputs, (tuple, list)):
            for output in outputs:
                yield from Graph._iter_output_variables(output)
            return
        raise TypeError(
            "Graph.forward() must return a Variable or a tuple/list of Variables"
        )

    @staticmethod
    def _capture(
        computations: tuple[Computation, ...],
    ) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
        nodes = []
        edges = []
        visited = set()
        seen_edges = set()

        def add(node: Any) -> None:
            if node in visited:
                return
            visited.add(node)
            for edge in node._in_edges:
                if id(edge) not in seen_edges:
                    seen_edges.add(id(edge))
                    edges.append(edge)
            nodes.append(node)

        for computation in computations:
            for node in computation._nodes:
                add(node)
        return tuple(nodes), tuple(edges)

    @staticmethod
    def _function_values(function: Callable[..., Any]) -> Iterator[Any]:
        """Yield Variables and child Graphs referenced by a function body."""
        closure = getclosurevars(function)
        yield from closure.nonlocals.values()
        yield from closure.globals.values()

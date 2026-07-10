"""Callable, reusable computational graph functions.

``Graph`` is the opt-in model abstraction.  A subclass defines ``forward``;
passing a function to ``Graph`` creates the equivalent functional model and
allows ``@Graph`` decorator use.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any
from inspect import getclosurevars

from ..autograd.variable import Variable
from .execution import replay


class Graph:
    """A reusable differentiable function.

    The first call executes ``forward`` eagerly and records the Variables
    reachable from its output.  Later compatible calls bind new input data and
    replay those recorded operations.  A subclass implements ``forward``;
    ``Graph(function)`` and ``@Graph`` provide the functional form.
    """

    def __init__(self, function: Callable[..., Any] | None = None) -> None:
        if function is not None and not callable(function):
            raise TypeError("Graph expects a callable function or no argument")

        self._function = function
        self._traced = False
        self._input_vars = ()
        self._input_keywords = ()
        self._input_shapes = ()
        self._outputs = None
        self._nodes = ()
        self._edges = ()

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Define a subclass model's eager computation."""
        if self._function is None:
            raise NotImplementedError("Graph subclasses must implement forward()")
        return self._function(*args, **kwargs)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Trace on the first call and replay on later compatible calls."""
        if not self._traced:
            return self._trace(args, kwargs)

        self._bind_inputs(args, kwargs)
        self._replay()
        return self._outputs

    def rebuild(self, *args: Any, **kwargs: Any) -> Any:
        """Discard the current trace and eagerly construct a new one."""
        self._traced = False
        self._input_vars = ()
        self._input_keywords = ()
        self._input_shapes = ()
        self._outputs = None
        self._nodes = ()
        self._edges = ()
        return self._trace(args, kwargs)

    @property
    def nodes(self) -> list[Any]:
        """Nodes reachable from this graph's output variables."""
        return list(self._nodes)

    @property
    def edges(self) -> list[Any]:
        """Edges reachable from this graph's output variables."""
        return list(self._edges)

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

        self._input_vars = input_args + tuple(input_kwargs[name] for name in keyword_names)
        self._input_keywords = keyword_names
        self._input_shapes = tuple(variable.shape for variable in self._input_vars)
        self._outputs = outputs
        self._nodes, self._edges = self._capture(output_vars)
        self._traced = True
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

    def _bind_inputs(self, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        if len(args) + len(kwargs) != len(self._input_vars):
            raise TypeError(
                f"Graph expects {len(self._input_vars)} inputs, got {len(args) + len(kwargs)}"
            )
        if set(kwargs) != set(self._input_keywords):
            expected = ", ".join(self._input_keywords) or "no keyword inputs"
            raise TypeError(f"Graph keyword inputs must be {expected}")

        values = list(args) + [kwargs[name] for name in self._input_keywords]
        for index, (variable, expected_shape, value) in enumerate(
            zip(self._input_vars, self._input_shapes, values)
        ):
            if isinstance(value, Variable):
                value = value.data
            data = variable.data.__class__(value)
            if data.shape != expected_shape:
                raise ValueError(
                    f"Graph input {index} has shape {data.shape}; expected "
                    f"{expected_shape}. Call rebuild() to trace a new shape."
                )
            variable.data = data

    def _replay(self) -> None:
        for output in self._iter_output_variables(self._outputs):
            replay(output)

    @staticmethod
    def _capture(outputs: tuple[Variable, ...]) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
        nodes = []
        edges = []
        visited = set()
        seen_edges = set()

        def visit(node: Any) -> None:
            if node in visited:
                return
            visited.add(node)
            for edge in node._in_edges:
                visit(edge.source)
                if id(edge) not in seen_edges:
                    seen_edges.add(id(edge))
                    edges.append(edge)
            nodes.append(node)

        for output in outputs:
            visit(output.node)
        return tuple(nodes), tuple(edges)

    @staticmethod
    def _function_values(function: Callable[..., Any]) -> Iterator[Any]:
        """Yield Variables and child Graphs referenced by a function body."""
        closure = getclosurevars(function)
        yield from closure.nonlocals.values()
        yield from closure.globals.values()

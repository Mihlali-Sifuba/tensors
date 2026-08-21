"""Callable, reusable computational graph functions.

``Graph`` is the opt-in model abstraction.  A subclass defines ``forward``;
passing a function to ``Graph`` creates the equivalent functional model and
allows ``@Graph`` decorator use.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from functools import partial
import threading
from typing import Any
from inspect import getclosurevars, isfunction, ismethod

from ..tensor import Tensor
from ..variable import Variable
from .computation import Computation
from .state import TraceScope


@dataclass
class GraphExecutionState:
    """Latest execution metadata for one Graph on one thread."""

    outputs: Any = None
    computations: tuple[Computation, ...] = ()
    nodes: tuple[Any, ...] = ()
    edges: tuple[Any, ...] = ()


class GraphThreadState(threading.local):
    """Thread-local storage with a typed Graph execution state."""

    execution: GraphExecutionState | None

    def __init__(self) -> None:
        self.execution = None


class Graph:
    """A callable differentiable model that records its latest computation.

    Every call executes ``forward`` eagerly and captures the Variables
    reachable from that call's output.  A subclass implements ``forward``;
    ``Graph(function)`` and ``@Graph`` provide the functional form.

    Execution metadata is kept per thread, so concurrent callers do not
    overwrite one another's latest computation.  Model parameters and other
    user-owned mutable attributes remain shared by the normal Python object
    rules and are not synchronized by this class.
    """

    def __init__(self, function: Callable[..., Any] | None = None) -> None:
        if function is not None and not callable(function):
            raise TypeError("Graph expects a callable function or no argument")

        self._function = function
        self._thread_state = GraphThreadState()

    def _state(self) -> GraphExecutionState:
        """Return this thread's latest execution metadata container."""
        state = self._thread_state.execution
        if state is None:
            state = GraphExecutionState()
            self._thread_state.execution = state
        return state

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
        The Graph can be called again to record a new computation.  Release
        applies to the calling thread's latest execution.
        """
        state = self._state()
        state.outputs = None
        state.computations = ()
        state.nodes = ()
        state.edges = ()

    @property
    def nodes(self) -> list[Any]:
        """Nodes reachable from this graph's output variables."""
        return list(self._state().nodes)

    @property
    def edges(self) -> list[Any]:
        """Edges reachable from this graph's output variables."""
        return list(self._state().edges)

    @property
    def computation(self) -> Computation:
        """Return the single computation produced by this graph."""
        computations = self._state().computations
        if not computations:
            raise RuntimeError("Graph has not been called yet")
        if len(computations) != 1:
            raise RuntimeError("Graph has multiple outputs; use computations")
        return computations[0]

    @property
    def computations(self) -> tuple[Computation, ...]:
        """Return computations for every flattened graph output."""
        return self._state().computations

    def parameters(self) -> list[Variable]:
        """Return persistent trainable Variables owned by this graph.

        Child Graph instances are traversed recursively.  Function graphs also
        inspect their closure so ``@Graph`` can capture simple parameters.
        """
        found: list[Variable] = []
        visited: set[int] = set()
        roots = [
            value
            for name, value in self._object_items(self)
            if name not in {"_function", "_thread_state"}
        ]
        if self._function is not None:
            roots.extend(self._function_values(self._function))

        # Use an explicit stack so deeply nested model containers do not
        # depend on Python's recursion limit. Reversing children preserves the
        # same depth-first discovery order as the former recursive traversal.
        pending = list(reversed(roots))
        while pending:
            value = pending.pop()
            identity = id(value)
            if identity in visited:
                continue
            visited.add(identity)

            if isinstance(value, Variable):
                if value.requires_grad:
                    found.append(value)
                continue

            children: list[Any] = []
            if isinstance(value, Graph):
                children.extend(
                    child
                    for name, child in self._object_items(value)
                    if name not in {"_function", "_thread_state"}
                )
                if value._function is not None:
                    children.extend(self._function_values(value._function))
            elif isinstance(value, dict):
                children.extend(value.values())
            elif isinstance(value, (list, tuple, set)):
                children.extend(value)
            elif (
                isfunction(value)
                or ismethod(value)
                or isinstance(value, partial)
                or (callable(value) and not isinstance(value, type))
            ):
                children.extend(self._function_values(value))
            pending.extend(reversed(children))
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

        state = self._state()
        state.outputs = outputs
        state.computations = tuple(Computation(output) for output in output_vars)
        state.nodes, state.edges = self._capture(state.computations)
        return outputs

    @staticmethod
    def _as_input(value: Any) -> Any:
        """Wrap Tensor inputs while preserving ordinary Python arguments."""
        if isinstance(value, Variable):
            return value
        if isinstance(value, Tensor):
            return Variable(value, requires_grad=False)
        return value

    @staticmethod
    def _iter_output_variables(outputs: Any) -> Iterator[Variable]:
        pending = [(outputs, False)]
        active_containers: set[int] = set()
        while pending:
            output, leaving = pending.pop()
            if leaving:
                active_containers.remove(id(output))
                continue
            if isinstance(output, Variable):
                yield output
                continue
            if not isinstance(output, (tuple, list)):
                raise TypeError(
                    "Graph.forward() must return a Variable or a tuple/list "
                    "of Variables"
                )

            identity = id(output)
            if identity in active_containers:
                raise ValueError("Graph outputs cannot contain cyclic containers")
            active_containers.add(identity)
            pending.append((output, True))
            pending.extend((item, False) for item in reversed(output))

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
        """Yield Values captured by a function or stored on a callable object."""
        if isinstance(function, partial):
            yield function.func
            yield from function.args
            yield from (function.keywords or {}).values()
            return

        if ismethod(function):
            yield from (
                value for _, value in Graph._object_items(function.__self__)
            )
            function = function.__func__

        if not isfunction(function):
            yield from (value for _, value in Graph._object_items(function))
            return

        yield from vars(function).values()
        closure = getclosurevars(function)
        yield from closure.nonlocals.values()
        yield from closure.globals.values()
        yield from (function.__defaults__ or ())
        yield from (function.__kwdefaults__ or {}).values()

    @staticmethod
    def _object_items(value: Any) -> Iterator[tuple[str, Any]]:
        """Yield stored attributes from dictionaries and ``__slots__``."""
        seen_names: set[str] = set()
        try:
            dictionary = vars(value)
        except TypeError:
            dictionary = {}
        for name, item in dictionary.items():
            seen_names.add(name)
            yield name, item

        for cls in type(value).__mro__:
            slots = cls.__dict__.get("__slots__", ())
            if isinstance(slots, str):
                slots = (slots,)
            for name in slots:
                if name in {"__dict__", "__weakref__"}:
                    continue
                storage_name = name
                if name.startswith("__") and not name.endswith("__"):
                    class_name = cls.__name__.lstrip("_")
                    storage_name = f"_{class_name}{name}"
                if storage_name in seen_names:
                    continue
                seen_names.add(storage_name)
                try:
                    yield storage_name, getattr(value, storage_name)
                except AttributeError:
                    continue

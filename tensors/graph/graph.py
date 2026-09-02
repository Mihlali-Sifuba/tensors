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


_UNCACHEABLE = object()


@dataclass(frozen=True, slots=True)
class _CompiledTrace:
    """One guarded trace and its input bindings for the current thread."""

    signature: Any
    positional_bindings: tuple[tuple[int, Variable], ...]
    keyword_bindings: tuple[tuple[str, Variable], ...]
    outputs: Any
    computations: tuple[Computation, ...]
    nodes: tuple[Any, ...]
    edges: tuple[Any, ...]
    structure_generation: int
    leaf_guards: tuple[tuple[Variable, Any, Any, bool], ...]


@dataclass(slots=True)
class GraphExecutionState:
    """Latest execution metadata for one Graph on one thread."""

    outputs: Any = None
    computations: tuple[Computation, ...] = ()
    nodes: tuple[Any, ...] = ()
    edges: tuple[Any, ...] = ()
    pending_outputs: tuple[Variable, ...] = ()
    pending_boundaries: tuple[Variable, ...] = ()
    compile_enabled: bool = False
    compiled: _CompiledTrace | None = None


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

    def __setattr__(self, name: str, value: Any) -> None:
        object.__setattr__(self, name, value)
        if name not in {
            "_function",
            "_thread_state",
            "_structure_generation",
            "_parameter_cache",
            "_parameter_graph_generations",
        }:
            generation = getattr(self, "_structure_generation", 0)
            object.__setattr__(self, "_structure_generation", generation + 1)
            object.__setattr__(self, "_parameter_cache", None)
            object.__setattr__(self, "_parameter_graph_generations", ())

    def __init__(self, function: Callable[..., Any] | None = None) -> None:
        if function is not None and not callable(function):
            raise TypeError("Graph expects a callable function or no argument")

        object.__setattr__(self, "_structure_generation", 0)
        object.__setattr__(self, "_function", function)
        object.__setattr__(self, "_thread_state", GraphThreadState())
        object.__setattr__(self, "_parameter_cache", None)
        object.__setattr__(self, "_parameter_graph_generations", ())

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
        """Execute eagerly, or replay a matching explicitly compiled trace."""
        return self._execute(args, kwargs)

    def _execute(
        self,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        *,
        force_retrace: bool = False,
        require_cacheable: bool = False,
    ) -> Any:
        scope = TraceScope()
        try:
            state = self._state()
            signature = (
                self._call_signature(args, kwargs)
                if scope.outermost and state.compile_enabled
                else None
            )
            if require_cacheable and signature is None:
                raise TypeError(
                    "Compiled Graph inputs must be Tensors or stable Python "
                    "configuration values; Variable inputs require a fresh trace"
                )
            if (
                not force_retrace
                and signature is not None
                and state.compiled is not None
                and state.compiled.signature == signature
                and self._compiled_trace_is_valid(state.compiled)
            ):
                return self._replay_compiled(state, state.compiled, args, kwargs)
            return self._trace(
                args,
                kwargs,
                lazy=not scope.outermost,
                compiled_signature=signature,
            )
        finally:
            scope.close()

    def rebuild(self, *args: Any, **kwargs: Any) -> Any:
        """Execute ``forward`` and record a fresh computation."""
        return self._execute(args, kwargs, force_retrace=True)

    def compile(self, *args: Any, **kwargs: Any) -> Any:
        """Trace once and enable guarded replay for compatible Tensor calls.

        The returned Variables are reused on cache hits and receive freshly
        replayed Tensor values. Calls with a different backend, Tensor shape,
        dtype, keyword layout, or static argument retrace and refresh the cache.
        Variable inputs always retain fresh-trace autograd semantics.
        """
        state = self._state()
        state.compile_enabled = True
        try:
            return self._execute(
                args,
                kwargs,
                force_retrace=True,
                require_cacheable=True,
            )
        except BaseException:
            state.compile_enabled = False
            state.compiled = None
            raise

    def uncompile(self) -> None:
        """Disable guarded replay while preserving latest execution metadata."""
        state = self._state()
        state.compile_enabled = False
        state.compiled = None

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
        state.pending_outputs = ()
        state.pending_boundaries = ()
        state.compile_enabled = False
        state.compiled = None

    @property
    def nodes(self) -> list[Any]:
        """Nodes reachable from this graph's output variables."""
        return list(self._materialize_state().nodes)

    @property
    def edges(self) -> list[Any]:
        """Edges reachable from this graph's output variables."""
        return list(self._materialize_state().edges)

    @property
    def computation(self) -> Computation:
        """Return the single computation produced by this graph."""
        computations = self._materialize_state().computations
        if not computations:
            raise RuntimeError("Graph has not been called yet")
        if len(computations) != 1:
            raise RuntimeError("Graph has multiple outputs; use computations")
        return computations[0]

    @property
    def computations(self) -> tuple[Computation, ...]:
        """Return computations for every flattened graph output."""
        return self._materialize_state().computations

    def parameters(self) -> list[Variable]:
        """Return persistent trainable Variables owned by this graph.

        Child Graph instances are traversed recursively.  Function graphs also
        inspect their closure so ``@Graph`` can capture simple parameters.
        """
        cached = self._parameter_cache
        if cached is not None and all(
            graph._structure_generation == generation
            for graph, generation in self._parameter_graph_generations
        ):
            return [variable for variable in cached if variable.requires_grad]

        found: list[Variable] = []
        graphs = [self]
        cacheable = self._function is None
        visited: set[int] = set()
        roots = [
            value
            for name, value in self._object_items(self)
            if name not in {
                "_function",
                "_thread_state",
                "_structure_generation",
                "_parameter_cache",
                "_parameter_graph_generations",
            }
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
                found.append(value)
                continue

            children: list[Any] = []
            if isinstance(value, Graph):
                graphs.append(value)
                if value._function is not None:
                    cacheable = False
                children.extend(
                    child
                    for name, child in self._object_items(value)
                    if name not in {
                        "_function",
                        "_thread_state",
                        "_structure_generation",
                        "_parameter_cache",
                        "_parameter_graph_generations",
                    }
                )
                if value._function is not None:
                    children.extend(self._function_values(value._function))
            elif isinstance(value, dict):
                cacheable = False
                children.extend(value.values())
            elif isinstance(value, (list, set)):
                cacheable = False
                children.extend(value)
            elif isinstance(value, tuple):
                children.extend(value)
            elif (
                isfunction(value)
                or ismethod(value)
                or isinstance(value, partial)
                or (callable(value) and not isinstance(value, type))
            ):
                cacheable = False
                children.extend(self._function_values(value))
            pending.extend(reversed(children))
        if cacheable:
            object.__setattr__(self, "_parameter_cache", tuple(found))
            object.__setattr__(
                self,
                "_parameter_graph_generations",
                tuple(
                    (graph, graph._structure_generation)
                    for graph in graphs
                ),
            )
        return [variable for variable in found if variable.requires_grad]

    def invalidate_parameters(self) -> None:
        """Drop cached structural parameter discovery for this graph."""
        object.__setattr__(self, "_parameter_cache", None)
        object.__setattr__(self, "_parameter_graph_generations", ())

    def _trace(
        self,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        *,
        lazy: bool = False,
        compiled_signature: Any = None,
    ) -> Any:
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
        boundaries = tuple(
            value
            for value in (*input_args, *input_kwargs.values())
            if isinstance(value, Variable)
        )
        if lazy:
            state.computations = ()
            state.nodes = ()
            state.edges = ()
            state.pending_outputs = output_vars
            state.pending_boundaries = boundaries
        else:
            self._record_state(state, output_vars, boundaries)
            if compiled_signature is not None:
                state.compiled = _CompiledTrace(
                    signature=compiled_signature,
                    positional_bindings=tuple(
                        (index, input_args[index])
                        for index, value in enumerate(args)
                        if isinstance(value, Tensor)
                    ),
                    keyword_bindings=tuple(
                        (name, input_kwargs[name])
                        for name in keyword_names
                        if isinstance(kwargs[name], Tensor)
                    ),
                    outputs=outputs,
                    computations=state.computations,
                    nodes=state.nodes,
                    edges=state.edges,
                    structure_generation=self._structure_generation,
                    leaf_guards=tuple(
                        (
                            variable,
                            variable.shape,
                            variable.dtype,
                            variable.requires_grad,
                        )
                        for _, variable in state.computations[0]._leaf_slots
                    ),
                )
        return outputs

    @staticmethod
    def _replay_compiled(
        state: GraphExecutionState,
        compiled: _CompiledTrace,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        """Rebind Tensor leaves and execute a previously guarded plan once."""
        for index, variable in compiled.positional_bindings:
            value = args[index]
            if variable.data is not value:
                variable.data = value
        for name, variable in compiled.keyword_bindings:
            value = kwargs[name]
            if variable.data is not value:
                variable.data = value
        compiled.computations[0].forward()
        state.outputs = compiled.outputs
        state.computations = compiled.computations
        state.nodes = compiled.nodes
        state.edges = compiled.edges
        state.pending_outputs = ()
        state.pending_boundaries = ()
        return compiled.outputs

    def _compiled_trace_is_valid(self, compiled: _CompiledTrace) -> bool:
        """Reject released plans and structural changes not covered by inputs."""
        if (
            compiled.structure_generation != self._structure_generation
            or not compiled.computations
            or compiled.computations[0]._released
        ):
            return False
        return all(
            variable.shape == shape
            and variable.dtype == dtype
            and variable.requires_grad == requires_grad
            for variable, shape, dtype, requires_grad in compiled.leaf_guards
        )

    @classmethod
    def _call_signature(
        cls,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any | None:
        """Return a conservative replay guard for cacheable public inputs."""
        from ..backend import get_backend

        positional = []
        for value in args:
            signature = cls._input_signature(value)
            if signature is _UNCACHEABLE:
                return None
            positional.append(signature)
        keywords = []
        for name, value in kwargs.items():
            signature = cls._input_signature(value)
            if signature is _UNCACHEABLE:
                return None
            keywords.append((name, signature))
        return get_backend(), tuple(positional), tuple(keywords)

    @classmethod
    def _input_signature(cls, value: Any) -> Any:
        if isinstance(value, Tensor):
            return "tensor", tuple(value.shape), value.dtype.typecode
        if isinstance(value, Variable):
            return _UNCACHEABLE
        return cls._static_signature(value)

    @classmethod
    def _static_signature(cls, value: Any) -> Any:
        if isinstance(value, (Tensor, Variable)):
            return _UNCACHEABLE
        if value is None or isinstance(value, (bool, int, float, str, bytes)):
            return "static", type(value), value
        if isinstance(value, tuple):
            items = tuple(cls._static_signature(item) for item in value)
            return _UNCACHEABLE if _UNCACHEABLE in items else ("tuple", items)
        if isinstance(value, list):
            items = tuple(cls._static_signature(item) for item in value)
            return _UNCACHEABLE if _UNCACHEABLE in items else ("list", items)
        try:
            hash(value)
        except TypeError:
            return _UNCACHEABLE
        return "static", type(value), value

    @staticmethod
    def _record_state(
        state: GraphExecutionState,
        outputs: tuple[Variable, ...],
        boundaries: tuple[Variable, ...],
    ) -> None:
        """Compile one shared plan and capture its structural metadata."""
        state.computations = Computation.from_outputs(
            outputs,
            boundaries=boundaries,
        )
        state.nodes, state.edges = Graph._capture(state.computations)
        state.pending_outputs = ()
        state.pending_boundaries = ()

    def _materialize_state(self) -> GraphExecutionState:
        """Compile the latest nested trace only when its metadata is requested."""
        state = self._state()
        if state.pending_outputs:
            self._record_state(
                state,
                state.pending_outputs,
                state.pending_boundaries,
            )
        return state

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
        if not computations:
            return (), ()
        nodes = computations[0]._all_nodes
        boundaries = computations[0]._boundary_nodes
        edges = tuple(
            edge
            for node in nodes
            if node not in boundaries
            for edge in node._in_edges
        )
        return nodes, edges

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

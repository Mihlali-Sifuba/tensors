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
from inspect import Parameter, getclosurevars, isfunction, ismethod, signature

from ..tensor import Tensor
from ..variable import Variable
from .computation import Computation
from .computation.compiler import (
    Compiler, resolve_boundaries, resolve_outputs,
)
from .expression import UnsupportedStructuralExpression
from .node import VariableNode
from .state import TraceScope, get_graph_state


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


@dataclass(frozen=True, slots=True)
class _ModelStructure:
    """A subclass model's structural graph and the program compiled from it.

    The graph is built once, from vertices standing in for the model's
    inputs, so it exists before any value does. Calling the model binds the
    input vertices to the call's Tensors and replays the program.
    """

    inputs: tuple[VariableNode, ...]
    outputs: Any
    computations: tuple[Computation, ...]
    nodes: tuple[Any, ...]
    edges: tuple[Any, ...]
    generation: int


class _GraphMeta(type):
    """Builds a subclass model's graph once its ``__init__`` has returned.

    A model's structure is written in terms of its parameters, so it can only
    be recorded after the subclass has finished creating them. Construction
    therefore ends with the build rather than deferring it to the first call.
    """

    def __call__(cls, *args: Any, **kwargs: Any) -> Any:
        instance = super().__call__(*args, **kwargs)
        if isinstance(instance, Graph):
            instance._build_model_structure()
        return instance


def _structural_value(value: Any) -> Variable:
    """Return the runtime Variable a structural output names."""
    if isinstance(value, VariableNode):
        return value.variable
    if isinstance(value, Variable):
        return value
    raise TypeError(
        "Graph.forward() must return a Variable or a tuple/list of Variables"
    )


def _structural_outputs(outputs: Any) -> Any:
    """Return ``outputs`` with every vertex replaced by the value it names.

    Containers are rebuilt with an explicit stack, so a deeply nested output
    does not depend on Python's recursion limit.
    """
    if not isinstance(outputs, (tuple, list)):
        return _structural_value(outputs)

    built: dict[int, Any] = {}
    pending: list[tuple[Any, bool]] = [(outputs, False)]
    while pending:
        item, expanded = pending.pop()
        if expanded:
            built[id(item)] = type(item)(
                built[id(child)]
                if isinstance(child, (tuple, list))
                else _structural_value(child)
                for child in item
            )
            continue
        pending.append((item, True))
        pending.extend(
            (child, False)
            for child in reversed(item)
            if isinstance(child, (tuple, list))
        )
    return built[id(outputs)]


def _iter_output_nodes(outputs: Any) -> Iterator[VariableNode]:
    """Yield the vertex naming each value a structural forward returned."""
    pending = [(outputs, False)]
    active_containers: set[int] = set()
    while pending:
        output, leaving = pending.pop()
        if leaving:
            active_containers.remove(id(output))
            continue
        if isinstance(output, VariableNode):
            yield output
            continue
        if isinstance(output, Variable):
            yield output.node
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


class GraphThreadState(threading.local):
    """Thread-local storage with a typed Graph execution state."""

    execution: GraphExecutionState | None

    def __init__(self) -> None:
        self.execution = None


class Graph(metaclass=_GraphMeta):
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
            "_structure",
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
        object.__setattr__(self, "_structure", None)
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

    # -- structural model construction ---------------------------------

    def _build_model_structure(self) -> None:
        """Build and compile this model's graph once ``__init__`` has run.

        A subclass states its computation over its own parameters, so the
        graph can be recorded as soon as those exist. Vertices stand in for
        the model's inputs, the structural ``forward`` records the topology
        without calculating anything, and the result is compiled into the
        program every call replays.

        A model that cannot be described before its values exist keeps the
        tracing lifecycle it had: a functional graph, a ``forward`` taking
        configuration arguments whose values are only known per call, and a
        ``forward`` whose expression the graph cannot record yet — a Python
        scalar operand, or a function that has no structural form. Only that
        last case is caught, and only through the signal that states it:
        anything else wrong with a model or with the machinery that records
        it is a real failure, and construction reports it here.
        """
        inputs = self._structural_input_count()
        if inputs is None:
            return
        try:
            structure = self._record_structure(inputs)
        except UnsupportedStructuralExpression:
            return
        object.__setattr__(self, "_structure", structure)

    def _structural_input_count(self) -> int | None:
        """Return how many inputs a structural build passes, if it can."""
        if self._function is not None or type(self).forward is Graph.forward:
            return None
        parameters = list(signature(type(self).forward).parameters.values())
        count = 0
        for parameter in parameters[1:]:
            if parameter.kind not in (
                Parameter.POSITIONAL_ONLY,
                Parameter.POSITIONAL_OR_KEYWORD,
            ) or parameter.default is not Parameter.empty:
                # A configuration argument is only known per call, so the
                # model keeps tracing rather than guessing a value for it.
                return None
            count += 1
        return count or None

    def _record_structure(self, inputs: int) -> _ModelStructure:
        """Record and compile the graph of one structural forward pass."""
        generation = self._structure_generation
        graph = get_graph_state()
        input_nodes = tuple(graph.add_variable_node() for _ in range(inputs))
        outputs = self.forward(*input_nodes)
        output_nodes = tuple(_iter_output_nodes(outputs))
        if not output_nodes:
            raise TypeError(
                "Graph.forward() must return a Variable or a tuple/list of "
                "Variables"
            )
        compiler = Compiler(output_nodes, boundaries=input_nodes)
        compiler.compile()
        return _ModelStructure(
            inputs=input_nodes,
            outputs=outputs,
            computations=Computation._from_compiler(compiler),
            nodes=compiler.nodes,
            edges=compiler.edges,
            generation=generation,
        )

    def _structure_for_replay(
        self,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> _ModelStructure | None:
        """Return the built structure this call can replay, if any.

        A Variable input keeps the tracing lifecycle, because its autograd
        identity belongs to the caller and a built input vertex already names
        a value of its own.
        """
        structure = self._structure
        if (
            structure is None
            or kwargs
            or len(args) != len(structure.inputs)
            or not all(isinstance(value, Tensor) for value in args)
        ):
            return None
        if structure.generation != self._structure_generation or any(
            computation._released for computation in structure.computations
        ):
            structure = self._record_structure(len(structure.inputs))
            object.__setattr__(self, "_structure", structure)
        return structure

    def _replay_structure(
        self,
        structure: _ModelStructure,
        args: tuple[Any, ...],
    ) -> Any:
        """Bind this call's inputs to the built graph and replay its program."""
        for node, value in zip(structure.inputs, args):
            if node.is_bound:
                variable = node.variable
                if variable.data is not value:
                    variable.data = value
            else:
                node.materialize(value, requires_grad=False)
        for computation in structure.computations:
            computation.forward()

        state = self._state()
        state.outputs = _structural_outputs(structure.outputs)
        state.computations = structure.computations
        state.nodes = structure.nodes
        state.edges = structure.edges
        state.pending_outputs = ()
        state.pending_boundaries = ()
        return state.outputs

    def _apply_structurally(
        self,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        """Record this model's operations into the graph being described.

        A model called with vertices is part of a larger structural
        expression, so its ``forward`` records into that graph and nothing
        executes.
        """
        return self.forward(*args, **kwargs)

    def _execute(
        self,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        *,
        force_retrace: bool = False,
        require_cacheable: bool = False,
    ) -> Any:
        if any(
            isinstance(value, VariableNode)
            for value in (*args, *kwargs.values())
        ):
            return self._apply_structurally(args, kwargs)
        if not force_retrace:
            structure = self._structure_for_replay(args, kwargs)
            if structure is not None:
                return self._replay_structure(structure, args)
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
                # Ordinary eager calls only need the output-owned incoming
                # edges. Building the reusable execution plan is deferred
                # until graph metadata is inspected. Explicit compilation
                # still materializes immediately because replay needs it.
                lazy=signature is None,
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
        object.__setattr__(self, "_structure", None)

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
                "_structure",
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
                        "_structure",
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
                        for variable in (
                            state.computations[0]._leaf_variables()
                        )
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
        """Compile one shared plan and keep its structural metadata.

        The compiler is the component that understands graph structure, so it
        supplies the nodes and edges directly. One compilation serves both
        sides: the structural record kept here and the computations that
        execute the program it emitted.
        """
        compiler = Compiler(
            resolve_outputs(outputs),
            boundaries=resolve_boundaries(boundaries),
        )
        compiler.compile()
        state.computations = Computation._from_compiler(compiler)
        state.nodes = compiler.nodes
        state.edges = compiler.edges
        state.pending_outputs = ()
        state.pending_boundaries = ()

    def _materialize_state(self) -> GraphExecutionState:
        """Compile the latest nested trace only when its metadata is requested."""
        state = self._state()
        structure = self._structure
        if (
            structure is not None
            and not state.computations
            and not state.pending_outputs
        ):
            # A built model owns its graph from construction, so its
            # metadata is available before the first call. A trace waiting
            # to be compiled describes a more recent call, so it wins.
            state.outputs = None
            state.computations = structure.computations
            state.nodes = structure.nodes
            state.edges = structure.edges
            return state
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

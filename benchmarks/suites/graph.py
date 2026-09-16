"""Graph construction, compilation, and replay across topologies.

Four things happen between writing an expression and getting a number, and
they are measured separately:

structural construction
    recording vertices and edges, with no values involved.
eager trace
    running the expression as Variables, which records *and* executes.
compilation
    turning a recorded graph into an instruction program, broken into the
    dependency resolution, slot assignment, and instruction emission that
    the compiler performs in that order.
replay
    running the compiled program again, forwards or in reverse.

Depth is only one axis. A chain, a wide fan-in, a diamond, and a shared
subexpression compile to the same number of instructions but exercise
different traversal and reachability work, so topology is varied as well.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import tensors as ts
from tensors.graph import Computation
from tensors.graph.computation.compiler import Compiler
from tensors.graph.node import VariableNode
from tensors.graph.state import get_graph_state, isolated_graph_state
from tensors.ops import Add, Mul

from ..harness import Case, Group, Unsupported
from ..workloads import tensor


#: How many nodes a topology is built at.
NODE_COUNTS: tuple[int, ...] = (1, 10, 100, 1_000, 4_000)

#: Tensor widths, so per-node cost can be separated from per-element cost.
WIDTHS: tuple[int, ...] = (1, 1_000, 100_000)


# -- structural builders -------------------------------------------------
#
# Each builder records a topology over vertices without calculating
# anything, and returns the output vertex and the leaves it reads.


def _chain_nodes(count: int) -> tuple[VariableNode, tuple[VariableNode, ...]]:
    """Record a linear chain of ``count`` additions."""
    state = get_graph_state()
    leaf = state.add_variable_node()
    current = leaf
    for _ in range(count):
        current = state.record_operation(Add(), (current, leaf))
    return current, (leaf,)


def _wide_nodes(count: int) -> tuple[VariableNode, tuple[VariableNode, ...]]:
    """Record ``count`` independent operations summed into one output."""
    state = get_graph_state()
    leaf = state.add_variable_node()
    branches = [
        state.record_operation(Mul(), (leaf, leaf)) for _ in range(count)
    ]
    current = branches[0]
    for branch in branches[1:]:
        current = state.record_operation(Add(), (current, branch))
    return current, (leaf,)


def _diamond_nodes(count: int) -> tuple[VariableNode, tuple[VariableNode, ...]]:
    """Record repeated split-and-rejoin diamonds."""
    state = get_graph_state()
    leaf = state.add_variable_node()
    current = leaf
    for _ in range(max(count // 3, 1)):
        left = state.record_operation(Mul(), (current, current))
        right = state.record_operation(Add(), (current, current))
        current = state.record_operation(Add(), (left, right))
    return current, (leaf,)


def _shared_nodes(count: int) -> tuple[VariableNode, tuple[VariableNode, ...]]:
    """Record a chain in which one subexpression is reused throughout."""
    state = get_graph_state()
    leaf = state.add_variable_node()
    shared = state.record_operation(Mul(), (leaf, leaf))
    current = shared
    for _ in range(count):
        current = state.record_operation(Add(), (current, shared))
    return current, (leaf,)


TOPOLOGIES = {
    "chain": _chain_nodes,
    "wide": _wide_nodes,
    "diamond": _diamond_nodes,
    "shared": _shared_nodes,
}


def _structural_cases(
    backend: str, topology: str, count: int
) -> list[Case]:
    """Measure recording and compiling a topology, without values."""
    builder = TOPOLOGIES[topology]
    common: dict[str, Any] = {
        "family": f"graph/{topology}",
        "elements": count,
        "work_items": count,
        "gc_enabled": True,
        "tags": {
            "curve": f"graph-structure-{topology}",
            "topology": topology,
        },
    }
    cases: list[Case] = []

    def build() -> Any:
        # An isolated state keeps each construction from accumulating in the
        # thread's registry, which would otherwise make later samples
        # measure a growing registry rather than the construction.
        with isolated_graph_state():
            return builder(count)

    cases.append(Case(
        name=f"graph.construct/{topology}/{count}",
        run=build,
        layer="graph-trace",
        validate=build,
        description=(
            f"record a {topology} topology of {count} operations as "
            "structure only, calculating nothing"
        ),
        memory=True,
        **common,
    ))

    # Compilation, and the three phases it runs in order. Each phase case
    # repeats the phases before it, so the marginal cost of a phase is the
    # difference between consecutive cases.
    with isolated_graph_state():
        output, leaves = builder(count)

        def compile_all() -> Any:
            compiler = Compiler((output,))
            compiler.compile()
            return compiler

        def resolve_only() -> Any:
            compiler = Compiler((output,))
            compiler._resolve_dependencies()
            return compiler

        def resolve_and_slots() -> Any:
            compiler = Compiler((output,))
            compiler._resolve_dependencies()
            compiler._assign_slots()
            return compiler

        def resolve_slots_emit() -> Any:
            compiler = Compiler((output,))
            compiler._resolve_dependencies()
            compiler._assign_slots()
            compiler._emit_instructions()
            return compiler

        instruction_count = len(compile_all().instructions)

        for name, call, description in (
            (
                "dependencies",
                resolve_only,
                "traverse the graph and record per-output reachability",
            ),
            (
                "slots",
                resolve_and_slots,
                "dependency resolution plus slot assignment",
            ),
            (
                "instructions",
                resolve_slots_emit,
                "dependency resolution, slots, and instruction emission",
            ),
            (
                "compile",
                compile_all,
                "the complete compilation, including view resolution",
            ),
        ):
            compile_common = dict(common)
            compile_common["work_items"] = instruction_count or None
            compile_common["tags"] = {
                **common["tags"],
                "curve": f"graph-compile-{name}-{topology}",
                "phase": "compile",
            }
            cases.append(Case(
                name=f"graph.{name}/{topology}/{count}",
                run=call,
                layer="graph-compile",
                validate=call,
                description=description,
                **compile_common,
            ))

        metadata_common = dict(common)
        metadata_common["tags"] = {
            **common["tags"],
            "curve": f"graph-metadata-{topology}",
        }
        cases.append(Case(
            name=f"graph.edges_metadata/{topology}/{count}",
            run=lambda: Compiler((output,)).edges,
            layer="graph-compile",
            validate=lambda: Compiler((output,)).edges,
            description=(
                "materialize the structural edge metadata the graph layer "
                "keeps, which compilation itself does not calculate"
            ),
            **metadata_common,
        ))

    return cases


def _execution_cases(
    backend: str, topology: str, count: int, width: int
) -> list[Case]:
    """Compare Tensor, eager Variable, and replayed execution."""
    common: dict[str, Any] = {
        "family": f"graph/{topology}",
        "dtype": "float64",
        "shape": (width,),
        "elements": width,
        "work_items": count,
        "tags": {
            "curve": f"graph-execute-{topology}|{width}",
            "topology": topology,
            "pair": f"graph-execute-{topology}|{count}|{width}",
        },
    }
    cases: list[Case] = []
    value = tensor((width,), dtype_name="float64", kind="constant", value=1.0)

    # -- plain Tensor arithmetic: the same operations, no graph ---------
    if topology == "chain":
        def run_tensor() -> Any:
            current = value
            for _ in range(count):
                current = current + value
            return current
    else:
        def run_tensor() -> Any:
            current = value * value
            for _ in range(count):
                current = current + value
            return current

    cases.append(Case(
        name=f"tensor.execute/{topology}/{count}/{width}",
        run=run_tensor,
        layer="public",
        validate=run_tensor,
        description=(
            f"{count} plain Tensor operations, with no graph recorded"
        ),
        **common,
    ))

    # -- eager Variable arithmetic: records and executes ----------------
    variable = ts.Variable(value, requires_grad=False)

    if topology == "chain":
        def run_variable() -> Any:
            current = variable
            for _ in range(count):
                current = current + variable
            return current
    else:
        def run_variable() -> Any:
            current = variable * variable
            for _ in range(count):
                current = current + variable
            return current

    trace_common = dict(common)
    trace_common["tags"] = {**common["tags"], "phase": "trace"}
    cases.append(Case(
        name=f"variable.trace/{topology}/{count}/{width}",
        run=run_variable,
        layer="graph-trace",
        validate=run_variable,
        description=(
            f"{count} eager Variable operations, each recording structure "
            "and compiling and running a one-instruction program"
        ),
        gc_enabled=True,
        memory=True,
        **trace_common,
    ))

    # -- compiled replay ------------------------------------------------
    traced = run_variable()
    computation = Computation(traced)
    computation.forward()
    replay_common = dict(common)
    replay_common["tags"] = {**common["tags"], "phase": "replay"}
    cases.append(Case(
        name=f"graph.replay_forward/{topology}/{count}/{width}",
        run=computation.forward,
        layer="graph-replay",
        validate=computation.forward,
        description=f"forward replay of {count} compiled instructions",
        memory=True,
        **replay_common,
    ))

    # -- reverse replay -------------------------------------------------
    differentiable = ts.Variable(value, requires_grad=True)
    if topology == "chain":
        current = differentiable
        for _ in range(count):
            current = current + differentiable
    else:
        current = differentiable * differentiable
        for _ in range(count):
            current = current + differentiable
    reverse = Computation(current)
    reverse.forward()
    reverse.backward()

    backward_common = dict(common)
    backward_common["tags"] = {
        **common["tags"],
        "phase": "backward",
        "pair": f"graph-replay-{topology}|{count}|{width}",
    }
    forward_pair = dict(common)
    forward_pair["tags"] = {
        **common["tags"],
        "phase": "forward",
        "pair": f"graph-replay-{topology}|{count}|{width}",
    }
    cases.append(Case(
        name=f"graph.replay_forward_paired/{topology}/{count}/{width}",
        run=reverse.forward,
        layer="graph-replay",
        validate=reverse.forward,
        description="the forward replay its reverse pass is compared against",
        **forward_pair,
    ))
    cases.append(Case(
        name=f"graph.replay_backward/{topology}/{count}/{width}",
        run=reverse.backward,
        layer="autograd",
        validate=reverse.backward,
        description=f"reverse replay of {count} compiled instructions",
        memory=True,
        **backward_common,
    ))
    return cases


# -- Graph models --------------------------------------------------------


class _Linear(ts.Graph):
    """A single affine layer, as a structural Graph model."""

    def __init__(self, features: int, units: int) -> None:
        super().__init__()
        self.weight = ts.Variable(
            tensor((features, units), dtype_name="float64", kind="constant",
                   value=0.05),
            name="weight",
        )
        self.bias = ts.Variable(
            tensor((units,), dtype_name="float64", kind="constant", value=0.01),
            name="bias",
        )

    def forward(self, inputs: Any) -> Any:
        return ts.relu(inputs @ self.weight + self.bias)


class _Nested(ts.Graph):
    """A model built from two nested Graph models."""

    def __init__(self, features: int, units: int) -> None:
        super().__init__()
        self.first = _Linear(features, units)
        self.second = _Linear(units, units)

    def forward(self, inputs: Any) -> Any:
        return self.second(self.first(inputs))


def _model_cases(backend: str, batch: int, features: int) -> list[Case]:
    """Measure structural Graph models, including a nested one."""
    common: dict[str, Any] = {
        "family": "graph/model",
        "dtype": "float64",
        "shape": (batch, features),
        "elements": batch * features,
        "tags": {"curve": f"graph-model|{features}"},
    }
    cases: list[Case] = []
    inputs = tensor(
        (batch, features), dtype_name="float64", kind="constant", value=0.5
    )

    model = _Linear(features, features)
    model(inputs)
    cases.append(Case(
        name=f"graph.model_call/{batch}x{features}",
        run=lambda: model(inputs),
        layer="graph-replay",
        validate=lambda: model(inputs),
        description=(
            "call a structural Graph model, which binds inputs and replays "
            "the program built at construction"
        ),
        memory=True,
        **common,
    ))

    nested = _Nested(features, features)
    nested(inputs)
    cases.append(Case(
        name=f"graph.nested_model_call/{batch}x{features}",
        run=lambda: nested(inputs),
        layer="graph-replay",
        validate=lambda: nested(inputs),
        description="call a model composed of two nested Graph models",
        **common,
    ))

    def construct() -> Any:
        # Construction builds and compiles the model's graph, so this is
        # the cost of creating the model rather than of calling it.
        return _Linear(features, features)

    construct_common = dict(common)
    construct_common["tags"] = {
        **common["tags"], "curve": f"graph-model-construct|{features}",
    }
    cases.append(Case(
        name=f"graph.model_construct/{features}",
        run=construct,
        layer="graph-compile",
        validate=construct,
        description=(
            "construct a structural Graph model, which records and compiles "
            "its graph as construction ends"
        ),
        gc_enabled=True,
        memory=True,
        **construct_common,
    ))

    # A traced model, opted into guarded replay, against the same model
    # retracing every call.
    traced_model = ts.Graph(
        lambda values: ts.relu(values * 2.0) + 1.0
    )
    traced_model(inputs)
    trace_common = dict(common)
    trace_common["tags"] = {
        **common["tags"],
        "curve": f"graph-function-model|{features}",
        "pair": f"graph-function-model|{batch}x{features}",
        "phase": "trace",
    }
    cases.append(Case(
        name=f"graph.function_trace/{batch}x{features}",
        run=lambda: traced_model(inputs),
        layer="graph-trace",
        validate=lambda: traced_model(inputs),
        description="a functional Graph model that retraces on every call",
        gc_enabled=True,
        **trace_common,
    ))

    compiled_model = ts.Graph(
        lambda values: ts.relu(values * 2.0) + 1.0
    )
    compiled_model.compile(inputs)
    replay_common = dict(common)
    replay_common["tags"] = {
        **common["tags"],
        "curve": f"graph-function-model|{features}",
        "pair": f"graph-function-model|{batch}x{features}",
        "phase": "replay",
    }
    cases.append(Case(
        name=f"graph.function_replay/{batch}x{features}",
        run=lambda: compiled_model(inputs),
        layer="graph-replay",
        validate=lambda: compiled_model(inputs),
        description=(
            "the same functional model after compile(), replaying through "
            "its guard instead of retracing"
        ),
        **replay_common,
    ))
    return cases


def groups() -> list[Group]:
    """Return structural, execution, and model groups."""
    result: list[Group] = []

    for topology in TOPOLOGIES:
        for count in NODE_COUNTS:
            def structural(
                backend: str, topology: str = topology, count: int = count,
            ) -> Sequence[Case]:
                return _structural_cases(backend, topology, count)

            result.append(Group(
                name=f"graph/structure/{topology}/{count}",
                factory=structural,
                suite="graph",
            ))

    for topology in ("chain", "wide"):
        for count in (1, 10, 100, 1_000):
            for width in WIDTHS:
                def execution(
                    backend: str,
                    topology: str = topology,
                    count: int = count,
                    width: int = width,
                ) -> Sequence[Case]:
                    if backend == "python" and count * width > 100_000:
                        raise Unsupported(
                            f"{count} operations over {width} elements "
                            "exceeds the Python backend graph ceiling"
                        )
                    if count * width > 20_000_000:
                        raise Unsupported(
                            f"{count} operations over {width} elements "
                            "would allocate more than this run permits"
                        )
                    return _execution_cases(
                        backend, topology, count, width
                    )

                result.append(Group(
                    name=f"graph/execute/{topology}/{count}/{width}",
                    factory=execution,
                    suite="graph",
                ))

    for batch, features in ((1, 16), (32, 128), (128, 512)):
        def model(
            backend: str, batch: int = batch, features: int = features,
        ) -> Sequence[Case]:
            if backend == "python" and batch * features > 20_000:
                raise Unsupported(
                    "exceeds the Python backend model ceiling"
                )
            return _model_cases(backend, batch, features)

        result.append(Group(
            name=f"graph/model/{batch}x{features}",
            factory=model,
            suite="graph",
        ))

    return result


__all__ = ["groups"]

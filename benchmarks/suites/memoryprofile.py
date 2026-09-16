"""Memory characterization: what operations allocate, and what they keep.

Timing and allocation tracing perturb each other, so the harness measures
them in separate passes. These cases exist mainly for the allocation pass:
each one is marked ``memory``, so it is also run under ``tracemalloc`` and
the CuPy pool, and reports what one call allocates, how many allocations it
makes, and what a batch of calls fails to release.

Retention is the point. A per-call retention near zero means the operation
releases what it takes; a positive one that persists across a batch is
growth, and growth over a training loop is what eventually matters.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import tensors as ts
from tensors.graph import Computation
from tensors.graph.state import get_graph_state

from ..case import Case, Group, Unsupported
from ..workloads import tensor


def _operation_cases(backend: str, size: int) -> list[Case]:
    """Measure what ordinary operations allocate and retain."""
    common: dict[str, Any] = {
        "family": "memory/operation",
        "dtype": "float64",
        "shape": (size,),
        "elements": size,
        "memory": True,
        "tags": {"curve": f"memory-operation|{size}"},
    }
    value = tensor((size,), dtype_name="float64", kind="ramp")
    matrix_side = max(int(size ** 0.5), 2)
    matrix = tensor(
        (matrix_side, matrix_side), dtype_name="float64", kind="ramp"
    )

    definitions: dict[str, Any] = {
        "add": lambda: value + value,
        "exp": lambda: ts.exp(value),
        "sum": lambda: ts.sum(value),
        "mean": lambda: ts.mean(value),
        "std": lambda: ts.std(value),
        "matmul": lambda: ts.matmul(matrix, matrix),
        "transpose": lambda: ts.transpose(matrix),
        "reshape": lambda: ts.reshape(value, (size, 1)),
        "astype": lambda: value.astype(ts.float32),
        "tolist": lambda: value.tolist(),
        "clone": lambda: value.clone(),
    }
    return [
        Case(
            name=f"memory.{name}/{size}",
            run=call,
            layer="memory",
            validate=call,
            description=f"allocation behavior of {name}",
            **common,
        )
        for name, call in definitions.items()
    ]


def _graph_growth_cases(backend: str, depth: int, size: int) -> list[Case]:
    """Measure what eager tracing and replay retain.

    An eager Variable operation records a result vertex that references its
    Variable and vice versa, and appends a weak outgoing-edge reference to
    each operand's vertex. Whether repeated eager work grows without bound
    is therefore a real question, and this is where it is asked.
    """
    common: dict[str, Any] = {
        "family": "memory/graph",
        "dtype": "float64",
        "shape": (size,),
        "elements": size,
        "work_items": depth,
        "memory": True,
        "gc_enabled": True,
        "tags": {"curve": f"memory-graph|{size}"},
    }
    value = tensor((size,), dtype_name="float64", kind="constant", value=1.0)
    leaf = ts.Variable(value, requires_grad=False)

    def eager_chain() -> Any:
        current = leaf
        for _ in range(depth):
            current = current + leaf
        return current

    cases: list[Case] = [
        Case(
            name=f"memory.eager_trace/{depth}/{size}",
            run=eager_chain,
            layer="memory",
            validate=eager_chain,
            description=(
                f"a {depth}-operation eager chain, discarded afterwards; "
                "its retention says whether tracing accumulates"
            ),
            **common,
        ),
    ]

    # A persistent leaf accumulates one weak outgoing-edge reference per
    # operation performed on it, pruned only when the list is next read.
    persistent = ts.Variable(value, requires_grad=False)

    def single_operation() -> Any:
        return persistent + persistent

    cases.append(Case(
        name=f"memory.persistent_leaf_edges/{size}",
        run=single_operation,
        layer="memory",
        validate=single_operation,
        description=(
            "one eager operation on a long-lived leaf, repeated; retention "
            "here is growth on the leaf's outgoing-edge list"
        ),
        **{**common, "work_items": 1},
    ))

    traced = eager_chain()
    computation = Computation(traced)
    computation.forward()
    cases.append(Case(
        name=f"memory.replay/{depth}/{size}",
        run=computation.forward,
        layer="memory",
        validate=computation.forward,
        description=(
            "repeated replay of one compiled program, which should reuse "
            "its Variables and allocate only its results"
        ),
        **common,
    ))

    differentiable = ts.Variable(value, requires_grad=True)
    current = differentiable
    for _ in range(depth):
        current = current + differentiable
    reverse = Computation(ts.sum(current))
    reverse.forward()
    reverse.backward()
    cases.append(Case(
        name=f"memory.backward/{depth}/{size}",
        run=reverse.backward,
        layer="memory",
        validate=reverse.backward,
        description=(
            "repeated reverse passes over one graph, which allocate "
            "gradients and publish them onto the Variables"
        ),
        **common,
    ))

    def graph_state_size() -> int:
        return len(get_graph_state().nodes)

    cases.append(Case(
        name=f"memory.graph_state_nodes/{size}",
        run=graph_state_size,
        layer="memory",
        validate=graph_state_size,
        description=(
            "reading the thread's node registry, whose cost grows with how "
            "much live structure the registry holds"
        ),
        **{**common, "work_items": None, "memory": False},
    ))
    return cases


def _training_growth_cases(
    backend: str, batch: int, hidden: int
) -> list[Case]:
    """Measure retention across repeated training iterations."""
    from .training import MultiLayerPerceptron

    common: dict[str, Any] = {
        "family": "memory/training",
        "dtype": "float64",
        "shape": (batch, hidden),
        "elements": batch * hidden,
        "memory": True,
        "gc_enabled": True,
        "tags": {"curve": f"memory-training|{hidden}"},
    }
    model = MultiLayerPerceptron(hidden, hidden, 10, 2, "float64")
    optimizer = ts.optim.Adam(model.parameters(), learning_rate=0.001)
    inputs = tensor(
        (batch, hidden), dtype_name="float64", kind="constant", value=0.5
    )
    targets = tensor(
        (batch, 10), dtype_name="float64", kind="constant", value=1.0
    )

    def step() -> None:
        predictions = model(inputs)
        loss = ts.mean((predictions - targets) ** 2.0)
        optimizer.zero_grad()
        ts.backward(loss)
        optimizer.step()

    # A first step on fresh optimizer state, so state allocation is
    # attributed to it rather than spread over the loop.
    fresh: dict[str, Any] = {}

    def reset_fresh() -> None:
        fresh["model"] = MultiLayerPerceptron(
            hidden, hidden, 10, 2, "float64"
        )
        fresh["optimizer"] = ts.optim.Adam(
            fresh["model"].parameters(), learning_rate=0.001
        )

    def fresh_step() -> None:
        predictions = fresh["model"](inputs)
        loss = ts.mean((predictions - targets) ** 2.0)
        ts.backward(loss)
        fresh["optimizer"].step()

    return [
        Case(
            name=f"memory.training_first_step/{batch}x{hidden}",
            run=fresh_step,
            layer="memory",
            validate=fresh_step,
            reset=reset_fresh,
            single_shot=True,
            description=(
                "a first training step, including optimizer-state "
                "allocation"
            ),
            **common,
        ),
        Case(
            name=f"memory.training_steady_step/{batch}x{hidden}",
            run=step,
            layer="memory",
            validate=step,
            description=(
                "a steady training step, repeated; its retention is what "
                "would accumulate over an epoch"
            ),
            **common,
        ),
    ]


def _cache_cases(backend: str, size: int) -> list[Case]:
    """Measure what the storage representation cache holds."""
    common: dict[str, Any] = {
        "family": "memory/storage-cache",
        "dtype": "float64",
        "shape": (size,),
        "elements": size,
        "memory": True,
        "tags": {"curve": f"memory-storage-cache|{size}"},
    }

    holder: dict[str, ts.Tensor] = {}

    def reset() -> None:
        holder["value"] = tensor(
            (size,), dtype_name="float64", kind="ramp"
        )

    def populate_all() -> None:
        value = holder["value"]
        for kind in ts.available_backends():
            value._storage_for(kind)

    return [
        Case(
            name=f"memory.storage_cache_populate/{size}",
            run=populate_all,
            layer="memory",
            validate=populate_all,
            reset=reset,
            single_shot=True,
            description=(
                "populating every cached representation of one Tensor, so "
                "the same values are held once per backend"
            ),
            **common,
        ),
    ]


def groups() -> list[Group]:
    """Return the memory groups."""
    result: list[Group] = []

    for size in (1_000, 100_000, 1_000_000):
        def operations(backend: str, size: int = size) -> Sequence[Case]:
            if backend == "python" and size > 100_000:
                raise Unsupported(
                    "exceeds the Python backend memory-probe ceiling"
                )
            return _operation_cases(backend, size)

        result.append(Group(
            name=f"memory/operations/{size}",
            factory=operations,
            suite="memory",
        ))

        def caches(backend: str, size: int = size) -> Sequence[Case]:
            if backend == "python" and size > 100_000:
                raise Unsupported(
                    "exceeds the Python backend memory-probe ceiling"
                )
            return _cache_cases(backend, size)

        result.append(Group(
            name=f"memory/storage-cache/{size}",
            factory=caches,
            suite="memory",
        ))

    for depth in (10, 100):
        for size in (100, 10_000):
            def graph_growth(
                backend: str, depth: int = depth, size: int = size,
            ) -> Sequence[Case]:
                if backend == "python" and depth * size > 100_000:
                    raise Unsupported(
                        "exceeds the Python backend memory-probe ceiling"
                    )
                return _graph_growth_cases(backend, depth, size)

            result.append(Group(
                name=f"memory/graph/{depth}/{size}",
                factory=graph_growth,
                suite="memory",
            ))

    for batch, hidden in ((8, 64), (64, 256)):
        def training_growth(
            backend: str, batch: int = batch, hidden: int = hidden,
        ) -> Sequence[Case]:
            if backend == "python" and batch * hidden > 20_000:
                raise Unsupported(
                    "exceeds the Python backend memory-probe ceiling"
                )
            return _training_growth_cases(backend, batch, hidden)

        result.append(Group(
            name=f"memory/training/{batch}x{hidden}",
            factory=training_growth,
            suite="memory",
        ))
    return result


__all__ = ["groups"]

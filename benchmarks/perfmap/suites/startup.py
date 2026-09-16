"""Cold start, measured in fresh interpreters, and cold-versus-warm in process.

Steady-state numbers deliberately exclude first-call costs, so those costs
are measured here instead, and never mixed in.

A genuinely cold measurement needs a new interpreter, so each case runs one
and times it end to end. That includes interpreter startup, which is why
``startup.interpreter_baseline`` exists: subtract it to get the cost of what
the case added. The subtraction is left to the reader rather than baked in,
so the raw numbers stay interpretable.

The in-process cases measure the other kind of cold: a warm interpreter
whose caches have been cleared.
"""

from __future__ import annotations
import subprocess
import sys
from pathlib import Path
from typing import Any
import tensors as ts
from tensors.backend import loading
from tensors.backend.loading import load_backend
from benchmarks.perfmap.harness import Case, Group, Unsupported
from benchmarks.perfmap.workloads import ACCELERATED, CUDA_ONLY, tensor

_REPOSITORY = Path(__file__).resolve().parents[3]


def _run_script(source: str) -> None:
    """Run one statement in a fresh interpreter, failing loudly."""
    completed = subprocess.run(
        [sys.executable, "-c", source], cwd=_REPOSITORY, capture_output=True, text=True
    )
    if completed.returncode != 0:
        raise RuntimeError(f"startup script failed: {completed.stderr.strip()[:400]}")


COLD_CASES: tuple[tuple[str, str, frozenset[str] | None, str], ...] = (
    (
        "interpreter_baseline",
        "pass",
        None,
        "a fresh interpreter that does nothing; the baseline every other cold case includes",
    ),
    ("import_numpy", "import numpy", None, "importing NumPy in a fresh interpreter"),
    ("import_cupy", "import cupy", CUDA_ONLY, "importing CuPy in a fresh interpreter"),
    (
        "import_tensors",
        "import tensors",
        None,
        "importing the package, which imports no optional provider",
    ),
    (
        "available_backends",
        "import tensors; tensors.available_backends()",
        None,
        "probing installed backends, which imports CuPy and counts devices",
    ),
    (
        "first_tensor_construction",
        "import tensors; tensors.Tensor([1.0, 2.0])",
        None,
        "constructing the first Tensor",
    ),
    (
        "first_python_operation",
        "import tensors as ts\nwith ts.use_backend('python'):\n    a = ts.full((64,), 1.5)\n    a + a\n",
        None,
        "the first Python-backend operation",
    ),
    (
        "first_numpy_operation",
        "import tensors as ts\nwith ts.use_backend('numpy'):\n    a = ts.full((64,), 1.5)\n    a + a\n",
        None,
        "the first NumPy-backend operation, which imports NumPy and resolves a kernel",
    ),
    (
        "first_cuda_operation",
        "import tensors as ts\nwith ts.use_backend('cuda'):\n    a = ts.full((64,), 1.5)\n    a + a\n",
        CUDA_ONLY,
        "the first CUDA operation, which initializes a CUDA context and compiles or loads kernels",
    ),
    (
        "cuda_context_only",
        "import cupy; cupy.cuda.Device().synchronize()",
        CUDA_ONLY,
        "CUDA context initialization alone, without the package",
    ),
    (
        "first_graph_execution",
        "import tensors as ts\nwith ts.use_backend('numpy'):\n    x = ts.Variable(ts.full((64,), 1.5))\n    y = x * x\n    ts.graph.Computation(y).forward()\n",
        None,
        "the first traced and replayed graph",
    ),
    (
        "first_backward",
        "import tensors as ts\nwith ts.use_backend('numpy'):\n    x = ts.Variable(ts.full((64,), 1.5))\n    y = ts.sum(x * x)\n    ts.backward(y)\n",
        None,
        "the first reverse pass",
    ),
    (
        "first_training_step",
        "import tensors as ts\nwith ts.use_backend('numpy'):\n    w = ts.Variable(ts.full((16, 16), 0.1))\n    x = ts.full((8, 16), 0.5)\n    optimizer = ts.optim.Adam([w], learning_rate=0.01)\n    loss = ts.mean(ts.relu(x @ w) ** 2.0)\n    ts.backward(loss)\n    optimizer.step()\n",
        None,
        "a complete first training step from a cold interpreter",
    ),
)


def _cold_cases(backend: str) -> list[Case]:
    """Build the fresh-interpreter cases.

    These are properties of the environment rather than of the active
    backend, so they are measured once, under whichever backend is
    scheduled first, and declared as such.
    """
    cases: list[Case] = []
    for name, source, backends, description in COLD_CASES:
        if backends is not None and backend not in backends:
            continue
        cases.append(
            Case(
                name=f"startup.cold/{name}",
                run=lambda source=source: _run_script(source),
                layer="startup",
                validate=lambda source=source: _run_script(source),
                description=f"{description}; measured as the wall clock of a fresh interpreter, so it includes interpreter startup",
                family="startup/cold",
                elements=0,
                single_shot=True,
                backends=backends,
                tags={"curve": "startup-cold", "scope": "fresh-interpreter"},
            )
        )
    return cases


def _warm_cases(backend: str) -> list[Case]:
    """Compare cold and warm paths inside one live interpreter."""
    cases: list[Case] = []
    common: dict[str, Any] = {
        "family": "startup/warm",
        "elements": 0,
        "tags": {"curve": "startup-warm"},
    }
    value = tensor((1_024,), dtype_name="float64", kind="ramp")
    if backend in ACCELERATED:

        def cold_kernel_lookup() -> Any:
            loading.load_backend.cache_clear()
            return loading.load_backend(backend).add

        cases.append(
            Case(
                name="startup.kernel_lookup_cold",
                run=cold_kernel_lookup,
                layer="startup",
                validate=cold_kernel_lookup,
                description="construct and bind a provider after explicitly clearing the provider cache",
                backends=ACCELERATED,
                tags={
                    "curve": "startup-kernel-lookup",
                    "pair": "kernel-lookup",
                    "phase": "trace",
                },
                **{key: item for key, item in common.items() if key != "tags"},
            )
        )
        cases.append(
            Case(
                name="startup.kernel_lookup_warm",
                run=lambda: loading.load_backend(backend).add,
                layer="startup",
                validate=lambda: loading.load_backend(backend).add,
                description="a kernel lookup the cache already satisfies",
                backends=ACCELERATED,
                tags={
                    "curve": "startup-kernel-lookup",
                    "pair": "kernel-lookup",
                    "phase": "replay",
                },
                **{key: item for key, item in common.items() if key != "tags"},
            )
        )

        def cold_provider_import() -> Any:
            load_backend.cache_clear()
            return load_backend(backend)

        cases.append(
            Case(
                name="startup.provider_module_cold",
                run=cold_provider_import,
                layer="startup",
                validate=cold_provider_import,
                description="resolving the provider module after clearing its cache; the module stays in sys.modules, so this is the lookup rather than a real import",
                backends=ACCELERATED,
                tags={
                    "curve": "startup-provider-module",
                    "pair": "provider-module",
                    "phase": "trace",
                },
                **{key: item for key, item in common.items() if key != "tags"},
            )
        )
        cases.append(
            Case(
                name="startup.provider_module_warm",
                run=lambda: load_backend(backend),
                layer="startup",
                validate=lambda: load_backend(backend),
                description="the cached provider-module lookup",
                backends=ACCELERATED,
                tags={
                    "curve": "startup-provider-module",
                    "pair": "provider-module",
                    "phase": "replay",
                },
                **{key: item for key, item in common.items() if key != "tags"},
            )
        )

    def first_graph() -> Any:
        variable = ts.Variable(value, requires_grad=False)
        traced = variable * variable
        return ts.graph.Computation(traced).forward()

    warm_variable = ts.Variable(value, requires_grad=False)
    warm_traced = warm_variable * warm_variable
    warm_computation = ts.graph.Computation(warm_traced)
    warm_computation.forward()
    cases.append(
        Case(
            name="startup.graph_first_execution",
            run=first_graph,
            layer="startup",
            validate=first_graph,
            description="trace, compile, and run a graph from nothing, in a warm interpreter",
            gc_enabled=True,
            tags={
                "curve": "startup-graph",
                "pair": "graph-execution",
                "phase": "trace",
            },
            **{key: item for key, item in common.items() if key != "tags"},
        )
    )
    cases.append(
        Case(
            name="startup.graph_warm_replay",
            run=warm_computation.forward,
            layer="startup",
            validate=warm_computation.forward,
            description="replaying the already-compiled graph",
            tags={
                "curve": "startup-graph",
                "pair": "graph-execution",
                "phase": "replay",
            },
            **{key: item for key, item in common.items() if key != "tags"},
        )
    )
    return cases


def groups() -> list[Group]:
    """Return the cold-start and cold-versus-warm groups."""
    return [
        Group(name="startup/cold", factory=_cold_cases, suite="startup"),
        Group(name="startup/warm", factory=_warm_cases, suite="startup"),
    ]


__all__ = ["groups"]

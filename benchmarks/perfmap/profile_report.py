"""Run the targeted diagnostics and write them as JSON.

This is a companion to the benchmark run, not part of it. The benchmarks say
which paths are slow; these probes say what those paths do. Every number here
comes from instrumented execution and must not be compared against a timing.

Usage::

    python -m benchmarks.perfmap.profile_report --output out/profiling.json
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import tensors as ts

from .meta import environment_metadata
from .profiling import host_transfers, hot_functions, provider_calls
from .report import write_json
from .workloads import tensor


def _subjects(backend: str) -> dict[str, Callable[[], Any]]:
    """Return the operations worth attributing, by name.

    Sizes are large enough that per-element work is real and small enough
    that a profiled run stays quick; the counts a probe returns do not
    depend on the size anyway.
    """
    size = 100_000
    f64 = tensor((size,), dtype_name="float64", kind="ramp")
    f32 = tensor((size,), dtype_name="float32", kind="ramp")
    mixed = tensor((size,), dtype_name="float64", kind="mixed")
    i64 = tensor((size,), dtype_name="int64", kind="ramp")
    matrix = tensor((256, 256), dtype_name="float64", kind="ramp")
    matrix32 = tensor((256, 256), dtype_name="float32", kind="ramp")
    row = tensor((256,), dtype_name="float64", kind="ramp")
    small = tensor((1,), dtype_name="float64", kind="constant")

    variable = ts.Variable(f64, requires_grad=False)
    differentiable = ts.Variable(
        tensor((1_000,), dtype_name="float64", kind="constant", value=1.0)
    )
    chain = differentiable
    for _ in range(10):
        chain = chain * 1.0009
    chain_output = ts.sum(chain)
    chain_computation = ts.graph.Computation(chain_output)
    chain_computation.forward()

    subjects: dict[str, Callable[[], Any]] = {
        # Elementwise, both dtypes: the float32 range guard is the question.
        "add/float64": lambda: f64 + f64,
        "add/float32": lambda: f32 + f32,
        "multiply/float64": lambda: f64 * f64,
        "divide/float64": lambda: f64 / f64,
        "exp/float64": lambda: ts.exp(f64),
        "exp/float32": lambda: ts.exp(f32),
        "tanh/float64": lambda: ts.tanh(f64),
        "relu/float64": lambda: ts.relu(f64),
        # Integer arithmetic, which the CUDA kernels decline.
        "add/int64": lambda: i64 + i64,
        # Reductions, both data patterns: the stability guard is the question.
        "sum/float64/same-sign": lambda: ts.sum(f64),
        "sum/float64/mixed-sign": lambda: ts.sum(mixed),
        "sum/float32/same-sign": lambda: ts.sum(f32),
        "mean/float64": lambda: ts.mean(f64),
        "max/float64": lambda: ts.max(f64),
        "std/float64": lambda: ts.std(f64),
        "norm/float64": lambda: ts.norm(f64),
        "prod/float64": lambda: ts.prod(f64),
        # Matrix work, where the finite-result check is the question.
        "matmul/float64": lambda: ts.matmul(matrix, matrix),
        "matmul/float32": lambda: ts.matmul(matrix32, matrix32),
        "dot/float64": lambda: ts.dot(row, row),
        # Broadcasting.
        "broadcast_add/float64": lambda: matrix + row,
        # Shape work, to locate the copies.
        "transpose/float64": lambda: ts.transpose(matrix),
        "reshape/float64": lambda: ts.reshape(matrix, (256 * 256,)),
        "slice/float64": lambda: matrix[:128, :128],
        "astype/float64-to-float32": lambda: f64.astype(ts.float32),
        "contiguous/float64": lambda: matrix.contiguous(),
        # Storage and scalar extraction.
        "item/float64": lambda: small.item(),
        "index/float64": lambda: f64[0],
        "tolist/float64": lambda: f64.tolist(),
        # Creation.
        "zeros/float64": lambda: ts.zeros((size,)),
        "full/float64": lambda: ts.full((size,), 1.5),
        # Normalization and loss.
        "softmax/float64": lambda: ts.softmax(matrix, axis=-1),
        "cross_entropy/float64": lambda: ts.cross_entropy(
            matrix, ts.full((256, 256), 1.0 / 256)
        ),
        # The layers above public.
        "variable_add/float64": lambda: variable + variable,
        "graph_replay/chain-10": chain_computation.forward,
        "graph_backward/chain-10": chain_computation.backward,
        "optimizer_step/adam": None,
    }

    parameter = ts.Variable(
        tensor((size,), dtype_name="float64", kind="constant", value=0.5)
    )
    parameter.grad = tensor(
        (size,), dtype_name="float64", kind="constant", value=0.01
    )
    optimizer = ts.optim.Adam([parameter], learning_rate=0.001)
    optimizer.step()
    subjects["optimizer_step/adam"] = optimizer.step
    return subjects


def _tracing_subjects(backend: str) -> dict[str, Callable[[], Any]]:
    """Return graph-tracing operations, for per-node attribution."""
    from tensors.graph.computation.compiler import Compiler
    from tensors.graph.state import get_graph_state, isolated_graph_state
    from tensors.ops import Add

    scalar = tensor((1,), dtype_name="float64", kind="constant")
    left = ts.Variable(scalar, requires_grad=False)
    right = ts.Variable(scalar, requires_grad=False)

    with isolated_graph_state():
        state = get_graph_state()
        leaf = state.add_variable_node()
        current = leaf
        for _ in range(100):
            current = state.record_operation(Add(), (current, leaf))
        chain_output = current

        def compile_chain() -> Any:
            compiler = Compiler((chain_output,))
            compiler.compile()
            return compiler

        compile_chain()

    return {
        "eager_variable_operation": lambda: left + right,
        "record_operation": lambda: get_graph_state().record_operation(
            Add(), (left.node, right.node)
        ),
        "compile_chain_100": compile_chain,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m benchmarks.perfmap.profile_report",
        description=(
            "Run targeted diagnostics that attribute benchmark findings to "
            "provider calls, host transfers, and Python functions."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--backend",
        action="append",
        choices=("python", "numpy", "cuda"),
        help="backend to profile; repeatable, defaults to numpy and cuda",
    )
    parser.add_argument(
        "--profile-repeats",
        type=int,
        default=200,
        help="invocations per cProfile subject",
    )
    arguments = parser.parse_args(argv)

    backends = arguments.backend or [
        name for name in ts.available_backends() if name != "python"
    ]
    results: dict[str, Any] = {}

    for backend in backends:
        print(f"Profiling {backend}...", flush=True)
        with ts.use_backend(backend):
            subjects = _subjects(backend)
            per_backend: dict[str, Any] = {}
            for name, call in subjects.items():
                entry: dict[str, Any] = {}
                try:
                    entry["provider_calls"] = provider_calls(
                        call, backend=backend
                    )
                except Exception as error:  # noqa: BLE001 - reported
                    entry["provider_calls"] = {
                        "error": f"{type(error).__name__}: {error}"
                    }
                if backend == "cuda":
                    try:
                        entry["host_transfers"] = host_transfers(call)
                    except Exception as error:  # noqa: BLE001
                        entry["host_transfers"] = {
                            "error": f"{type(error).__name__}: {error}"
                        }
                per_backend[name] = entry
                calls = entry.get("provider_calls", {})
                transfers = entry.get("host_transfers", {})
                print(
                    f"  {name:34} "
                    f"provider={calls.get('total_calls', '-'):>4} "
                    f"transfers={transfers.get('total_transfers', '-')}",
                    flush=True,
                )
            results[backend] = {"operations": per_backend}

            # cProfile is expensive, so it is applied to the paths whose
            # fixed overhead the benchmarks flagged rather than to all.
            focus = (
                "add/float64", "add/float32", "sum/float64/same-sign",
                "matmul/float64", "variable_add/float64",
                "graph_replay/chain-10", "graph_backward/chain-10",
                "item/float64", "transpose/float64",
            )
            profiles: dict[str, Any] = {}
            for name in focus:
                call = subjects.get(name)
                if call is None:
                    continue
                try:
                    profiles[name] = hot_functions(
                        call, repeats=arguments.profile_repeats
                    )
                except Exception as error:  # noqa: BLE001
                    profiles[name] = {
                        "error": f"{type(error).__name__}: {error}"
                    }
            results[backend]["hot_functions"] = profiles

            tracing: dict[str, Any] = {}
            for name, call in _tracing_subjects(backend).items():
                try:
                    tracing[name] = hot_functions(call, repeats=200)
                except Exception as error:  # noqa: BLE001
                    tracing[name] = {
                        "error": f"{type(error).__name__}: {error}"
                    }
            results[backend]["tracing"] = tracing

    report = {
        "schema": "tensors-perfmap-profiling/1",
        "kind": "diagnostic profiling, not measurement",
        "metadata": environment_metadata(),
        "note": (
            "Provider-call and host-transfer counts come from instrumented "
            "execution: the provider module the kernels call is replaced by "
            "a counting proxy, and CuPy's host-conversion methods are "
            "wrapped. Timings inside cProfile results are inflated by the "
            "profiler and are only meaningful relative to one another."
        ),
        "backends": results,
    }
    write_json(report, arguments.output)
    print(f"\nWrote {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

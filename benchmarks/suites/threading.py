"""Concurrent training steps in separate thread-local graphs.

Tracing state is per-thread, so two threads can each record and differentiate
their own graph without seeing each other's. This measures whether that
isolation actually lets the work overlap: the concurrent case is compared
against the same two steps taken one after the other on this thread, so the
number to read is the ratio between them, not either one alone.

A step mutates optimizer state, so each sample starts from fresh models and
runs exactly once. Batching several steps into one calibrated sample would
make the measured inputs depend on how fast the backend is.
"""

from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import tensors as ts

from ..harness import Case, Group, Unsupported
from ..workloads import FLOAT_DTYPES, tensor
from .training import MultiLayerPerceptron


def _step(
    model: MultiLayerPerceptron,
    optimizer: ts.optim.Optimizer,
    inputs: ts.Tensor,
    targets: ts.Tensor,
) -> float:
    """Run one full training step and return its loss."""
    prediction = model(inputs)
    loss = ts.mean((prediction - targets) ** 2.0)
    optimizer.zero_grad()
    ts.backward(loss)
    optimizer.step()
    return float(loss.data.item())


def _threading_cases(
    backend: str, batch: int, dtype_name: str, workers: int = 2
) -> list[Case]:
    """Build the concurrent and sequential forms of the same two steps."""
    features, hidden, outputs = 4, 8, 2
    inputs = tensor(
        (batch, features), dtype_name=dtype_name, kind="constant", value=0.25
    )
    targets = tensor(
        (batch, outputs), dtype_name=dtype_name, kind="constant", value=0.1
    )

    def fresh() -> tuple[list[Any], list[Any]]:
        models = [
            MultiLayerPerceptron(features, hidden, outputs, 1, dtype_name)
            for _ in range(workers)
        ]
        optimizers = [
            ts.optim.SGD(model.parameters(), learning_rate=1e-4) for model in models
        ]
        return models, optimizers

    models, optimizers = fresh()
    pool = ThreadPoolExecutor(max_workers=workers)

    def reset() -> None:
        nonlocal models, optimizers
        models, optimizers = fresh()

    def teardown() -> None:
        pool.shutdown(wait=True)

    def concurrent() -> list[float]:
        futures = [
            pool.submit(_step, models[index], optimizers[index], inputs, targets)
            for index in range(workers)
        ]
        return [future.result() for future in futures]

    def sequential() -> list[float]:
        return [
            _step(models[index], optimizers[index], inputs, targets)
            for index in range(workers)
        ]

    common: dict[str, Any] = {
        "family": "threading",
        "layer": "training",
        "dtype": dtype_name,
        "shape": (batch, features),
        "elements": batch * features,
        "work_items": workers * batch,
        "gc_enabled": True,
        "single_shot": True,
        "reset": reset,
        "teardown": teardown,
    }

    def tags(mode: str) -> dict[str, str]:
        """Return the keys that make this case comparable along two axes.

        ``pair`` holds the dtype, so it joins the two modes at one dtype;
        ``dtype_pair`` holds the mode, so it joins the two dtypes within one
        mode. Without the mode in ``dtype_pair`` the dtype comparison falls
        back to the family and silently pairs a concurrent run against a
        sequential one.
        """
        return {
            "pair": f"threading|{dtype_name}|b{batch}|w{workers}",
            "dtype_pair": f"threading-{mode}|b{batch}|w{workers}",
            "mode": mode,
            "workers": str(workers),
        }

    def validate(run: Any) -> Any:
        def check() -> None:
            results = run()
            assert len(results) == workers
            assert all(value == value for value in results)  # not NaN

        return check

    return [
        Case(
            name=f"threading.concurrent_steps/{dtype_name}/b{batch}/w{workers}",
            run=concurrent,
            validate=validate(concurrent),
            description=(
                f"{workers} training steps run at once, each recording and "
                "differentiating its own thread-local graph"
            ),
            tags=tags("concurrent"),
            **common,
        ),
        Case(
            name=f"threading.sequential_steps/{dtype_name}/b{batch}/w{workers}",
            run=sequential,
            validate=validate(sequential),
            description=(
                f"the same {workers} steps on one thread, which is what the "
                "concurrent case has to beat"
            ),
            tags=tags("sequential"),
            **common,
        ),
    ]


def groups() -> list[Group]:
    """Return one group per (batch, dtype) pair."""
    result: list[Group] = []
    for batch in (16, 256):
        for dtype_name in FLOAT_DTYPES:

            def factory(
                backend: str, batch: int = batch, dtype_name: str = dtype_name
            ) -> Sequence[Case]:
                if backend == "cuda":
                    raise Unsupported(
                        "concurrent steps share one CUDA stream, so the "
                        "device serializes them and the comparison measures "
                        "stream contention rather than graph isolation"
                    )
                return _threading_cases(backend, batch, dtype_name)

            result.append(
                Group(
                    name=f"threading/steps/{dtype_name}/{batch}",
                    factory=factory,
                    suite="threading",
                )
            )
    return result


__all__ = ["groups"]

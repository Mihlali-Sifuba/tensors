"""End-to-end training, and the phases a step is made of.

A training step is measured twice: as four separate phases and as one
combined step. The phases are what attribution needs; the combined step is
what tells us whether the phase numbers add up, and it is the number a
microbenchmark improvement ultimately has to move.

Phase boundaries are real here. Forward and loss are separated because the
loss is where a reduction and a normalization enter; backward is a single
reverse pass over the whole graph; the optimizer step is separate again.
Each phase case runs from the same state, restored by ``reset``, so a phase
never measures the leftovers of the previous sample.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import tensors as ts

from ..harness import Case, Group, Unsupported
from ..workloads import FLOAT_DTYPES, dtype_of, tensor


class MultiLayerPerceptron(ts.Graph):
    """A configurable fully connected network over its own parameters."""

    def __init__(
        self,
        features: int,
        hidden: int,
        outputs: int,
        depth: int,
        dtype_name: str,
    ) -> None:
        super().__init__()
        sizes = [features] + [hidden] * depth + [outputs]
        weights = []
        biases = []
        for index, (fan_in, fan_out) in enumerate(zip(sizes, sizes[1:])):
            scale = (2.0 / fan_in) ** 0.5
            weights.append(ts.Variable(
                tensor(
                    (fan_in, fan_out), dtype_name=dtype_name,
                    kind="constant", value=scale,
                ),
                name=f"weight_{index}",
            ))
            biases.append(ts.Variable(
                tensor(
                    (fan_out,), dtype_name=dtype_name,
                    kind="constant", value=0.01,
                ),
                name=f"bias_{index}",
            ))
        # Assigned as tuples so the Graph metaclass finds them as parameters.
        self.weights = tuple(weights)
        self.biases = tuple(biases)
        self.depth = depth

    def forward(self, inputs: Any) -> Any:
        current = inputs
        last = len(self.weights) - 1
        for index, (weight, bias) in enumerate(
            zip(self.weights, self.biases)
        ):
            current = current @ weight + bias
            if index != last:
                current = ts.relu(current)
        return current


def _training_cases(
    backend: str,
    name: str,
    features: int,
    hidden: int,
    outputs: int,
    depth: int,
    batch: int,
    dtype_name: str,
) -> list[Case]:
    """Build phase and combined-step cases for one model configuration."""
    dtype = dtype_of(dtype_name)
    suffix = f"{name}/{dtype_name}/b{batch}/h{hidden}/d{depth}"
    parameter_count = sum(
        fan_in * fan_out + fan_out
        for fan_in, fan_out in zip(
            [features] + [hidden] * depth,
            [hidden] * depth + [outputs],
        )
    )
    common: dict[str, Any] = {
        "family": f"training/{name}",
        "dtype": dtype_name,
        "shape": (batch, features),
        "elements": batch * features,
        "work_items": parameter_count,
        "gc_enabled": True,
        "tags": {
            "curve": f"training-{name}|{dtype_name}",
            "dtype_pair": f"training-{name}|b{batch}|h{hidden}|d{depth}",
            "pair": f"training-{suffix}",
            "parameters": str(parameter_count),
        },
    }

    inputs = tensor(
        (batch, features), dtype_name=dtype_name, kind="constant", value=0.5
    )
    targets = tensor(
        (batch, outputs), dtype_name=dtype_name, kind="constant", value=1.0
    )

    model = MultiLayerPerceptron(
        features, hidden, outputs, depth, dtype_name
    )
    optimizer = ts.optim.Adam(model.parameters(), learning_rate=0.001)

    # A shared holder so the phase cases can hand intermediate values to
    # one another without recomputing them.
    state: dict[str, Any] = {}

    def reset() -> None:
        state["predictions"] = model(inputs)
        state["loss"] = ts.mean(
            (state["predictions"] - targets) ** 2.0
        )
        ts.backward(state["loss"])

    def run_forward() -> Any:
        return model(inputs)

    def run_loss() -> Any:
        return ts.mean((state["predictions"] - targets) ** 2.0)

    def run_backward() -> None:
        ts.backward(state["loss"])

    def run_optimizer() -> None:
        optimizer.step()

    def run_step() -> Any:
        predictions = model(inputs)
        loss = ts.mean((predictions - targets) ** 2.0)
        optimizer.zero_grad()
        ts.backward(loss)
        optimizer.step()
        return loss

    cases: list[Case] = []
    for phase, call, layer, description in (
        (
            "forward",
            run_forward,
            "training",
            "the model's forward pass alone",
        ),
        (
            "loss",
            run_loss,
            "training",
            "the loss over already-computed predictions",
        ),
        (
            "backward",
            run_backward,
            "autograd",
            "one reverse pass over the whole step's graph",
        ),
        (
            "optimizer",
            run_optimizer,
            "optimizer",
            "the optimizer step over already-computed gradients",
        ),
    ):
        phase_common = dict(common)
        phase_common["tags"] = {**common["tags"], "phase": phase}
        cases.append(Case(
            name=f"training.{phase}/{suffix}",
            run=call,
            layer=layer,
            validate=call,
            reset=reset,
            description=description,
            **phase_common,
        ))

    step_common = dict(common)
    step_common["tags"] = {**common["tags"], "phase": "step"}
    cases.append(Case(
        name=f"training.step/{suffix}",
        run=run_step,
        layer="training",
        validate=run_step,
        description=(
            "one complete training step: forward, loss, zero, backward, "
            "and optimizer"
        ),
        memory=True,
        **step_common,
    ))

    # Ten steps in one timed call, so accumulated per-iteration cost and
    # any growth across iterations are visible in a single number.
    def run_ten_steps() -> None:
        for _ in range(10):
            run_step()

    sustained_common = dict(common)
    sustained_common["work_items"] = 10
    sustained_common["tags"] = {
        **common["tags"],
        "phase": "sustained",
        "curve": f"training-sustained-{name}|{dtype_name}",
    }
    cases.append(Case(
        name=f"training.ten_steps/{suffix}",
        run=run_ten_steps,
        layer="training",
        validate=run_ten_steps,
        description="ten consecutive training steps in one timed call",
        memory=True,
        **sustained_common,
    ))
    return cases


def _eager_versus_compiled(
    backend: str, batch: int, hidden: int, dtype_name: str
) -> list[Case]:
    """Compare a retracing functional model against a compiled one."""
    inputs = tensor(
        (batch, hidden), dtype_name=dtype_name, kind="constant", value=0.5
    )
    weight = ts.Variable(
        tensor(
            (hidden, hidden), dtype_name=dtype_name,
            kind="constant", value=0.05,
        )
    )
    bias = ts.Variable(
        tensor(
            (hidden,), dtype_name=dtype_name, kind="constant", value=0.01
        )
    )

    def body(values: Any) -> Any:
        return ts.relu(values @ weight + bias)

    common: dict[str, Any] = {
        "family": "training/eager-versus-compiled",
        "dtype": dtype_name,
        "shape": (batch, hidden),
        "elements": batch * hidden,
        "gc_enabled": True,
        "tags": {
            "curve": f"training-execution-mode|{dtype_name}",
            "pair": f"training-execution-mode|{batch}x{hidden}|{dtype_name}",
        },
    }

    eager = ts.Graph(body)
    eager(inputs)
    compiled = ts.Graph(body)
    try:
        compiled.compile(inputs)
    except Exception as error:  # noqa: BLE001 - classified, not hidden
        # ``compile`` only accepts a trace it can guard and replay. A model
        # it declines has no compiled form to compare against, which is the
        # answer rather than a failure.
        raise Unsupported(
            "Graph.compile() declined this model, so there is no compiled "
            f"form to compare eager tracing against: "
            f"{type(error).__name__}: {error}"
        ) from error

    eager_common = dict(common)
    eager_common["tags"] = {**common["tags"], "phase": "trace"}
    compiled_common = dict(common)
    compiled_common["tags"] = {**common["tags"], "phase": "replay"}
    return [
        Case(
            name=f"training.eager_layer/{batch}x{hidden}/{dtype_name}",
            run=lambda: eager(inputs),
            layer="graph-trace",
            validate=lambda: eager(inputs),
            description="a layer through a model that retraces every call",
            **eager_common,
        ),
        Case(
            name=f"training.compiled_layer/{batch}x{hidden}/{dtype_name}",
            run=lambda: compiled(inputs),
            layer="graph-replay",
            validate=lambda: compiled(inputs),
            description="the same layer through a compiled, replaying model",
            **compiled_common,
        ),
    ]


#: Model shapes: a name, and the geometry it stands for.
MODELS: dict[str, dict[str, int]] = {
    "tiny": {"features": 4, "hidden": 8, "outputs": 1, "depth": 1},
    "small": {"features": 32, "hidden": 64, "outputs": 10, "depth": 1},
    "moderate": {"features": 128, "hidden": 256, "outputs": 10, "depth": 2},
    "deep": {"features": 128, "hidden": 128, "outputs": 10, "depth": 8},
    "wide": {"features": 512, "hidden": 1_024, "outputs": 10, "depth": 2},
}

BATCHES: tuple[int, ...] = (1, 8, 64, 256)


def groups() -> list[Group]:
    """Return training groups over model shape, batch size, and dtype."""
    result: list[Group] = []

    for name, geometry in MODELS.items():
        for batch in BATCHES:
            for dtype_name in FLOAT_DTYPES:
                # Roughly how much arithmetic one forward pass performs,
                # used only to gate a configuration by backend.
                work = batch * geometry["hidden"] * max(
                    geometry["features"], geometry["hidden"]
                ) * (geometry["depth"] + 1)

                def factory(
                    backend: str,
                    name: str = name,
                    geometry: dict[str, int] = geometry,
                    batch: int = batch,
                    dtype_name: str = dtype_name,
                    work: int = work,
                ) -> Sequence[Case]:
                    if backend == "python" and work > 200_000:
                        raise Unsupported(
                            f"about {work} multiply-accumulates per forward "
                            "pass exceeds the Python backend training "
                            "ceiling"
                        )
                    return _training_cases(
                        backend,
                        name,
                        geometry["features"],
                        geometry["hidden"],
                        geometry["outputs"],
                        geometry["depth"],
                        batch,
                        dtype_name,
                    )

                result.append(Group(
                    name=(
                        f"training/{name}/{dtype_name}/b{batch}"
                    ),
                    factory=factory,
                    suite="training",
                ))

    for batch, hidden in ((1, 32), (64, 256), (256, 1_024)):
        for dtype_name in FLOAT_DTYPES:
            def mode_factory(
                backend: str,
                batch: int = batch,
                hidden: int = hidden,
                dtype_name: str = dtype_name,
            ) -> Sequence[Case]:
                if backend == "python" and batch * hidden > 20_000:
                    raise Unsupported(
                        "exceeds the Python backend ceiling for this "
                        "comparison"
                    )
                return _eager_versus_compiled(
                    backend, batch, hidden, dtype_name
                )

            result.append(Group(
                name=(
                    f"training/execution-mode/{dtype_name}/{batch}x{hidden}"
                ),
                factory=mode_factory,
                suite="training",
            ))
    return result


__all__ = ["groups"]

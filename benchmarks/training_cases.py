"""End-to-end model training-step benchmarks."""

from __future__ import annotations

import math

import tensors as ts

from .runner import BenchmarkBackend, BenchmarkCase


_ACCELERATED = frozenset[BenchmarkBackend]({"numpy", "cuda"})


class MLP(ts.Graph):
    """Small explicit model used only for end-to-end benchmark coverage."""

    def __init__(self) -> None:
        super().__init__()
        self.input_to_hidden = ts.Variable(
            ts.full((4, 8), 0.05),
            name="input_to_hidden",
        )
        self.hidden_bias = ts.Variable(
            ts.full((8,), 0.01),
            name="hidden_bias",
        )
        self.hidden_to_output = ts.Variable(
            ts.full((8, 2), 0.05),
            name="hidden_to_output",
        )
        self.output_bias = ts.Variable(
            ts.zeros((2,)),
            name="output_bias",
        )

    def forward(self, inputs):
        hidden = ts.relu(
            inputs @ self.input_to_hidden + self.hidden_bias
        )
        return hidden @ self.hidden_to_output + self.output_bias


def core_cases() -> list[BenchmarkCase]:
    batch_size = 16
    inputs = ts.Tensor(
        [
            float((row + column) % 7) / 7.0
            for row in range(batch_size)
            for column in range(4)
        ],
        shape=(batch_size, 4),
    )
    targets = ts.full((batch_size, 2), 0.25)
    model = MLP()
    optimizer = ts.optim.SGD(
        model.parameters(),
        learning_rate=1e-4,
    )

    def training_step():
        prediction = model(inputs)
        loss = ts.mean((prediction - targets) ** 2.0)
        optimizer.zero_grad()
        ts.backward(loss)
        optimizer.step()
        return loss

    def validate() -> None:
        loss = training_step()
        assert loss.size == 1
        assert math.isfinite(float(loss.data.item()))
        assert len(model.parameters()) == 4

    return [
        BenchmarkCase(
            name="training.mlp_step/batch-16",
            run=training_step,
            validate=validate,
            work_items=batch_size,
            gc_enabled=True,
            description=(
                "fresh model trace, MSE loss, backward pass, and SGD update"
            ),
            layer="training",
        )
    ]


class SizedMLP(ts.Graph):
    """Configurable matrix-heavy MLP for accelerated training measurements."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
    ) -> None:
        super().__init__()
        self.input_to_hidden = ts.Variable(
            ts.full((input_size, hidden_size), 0.01),
            name="input_to_hidden",
        )
        self.hidden_bias = ts.Variable(
            ts.full((hidden_size,), 0.01),
            name="hidden_bias",
        )
        self.hidden_to_output = ts.Variable(
            ts.full((hidden_size, output_size), 0.01),
            name="hidden_to_output",
        )
        self.output_bias = ts.Variable(
            ts.full((output_size,), 0.01),
            name="output_bias",
        )

    def forward(self, inputs):
        hidden = ts.relu(
            inputs @ self.input_to_hidden + self.hidden_bias
        )
        return hidden @ self.hidden_to_output + self.output_bias


def _medium_cases() -> list[BenchmarkCase]:
    batch_size = 64
    input_size = 32
    hidden_size = 64
    output_size = 16
    forward_work = batch_size * (
        input_size * hidden_size + hidden_size * output_size
    )
    inputs = ts.full((batch_size, input_size), 0.25)
    targets = ts.full((batch_size, output_size), 0.1)
    benchmarks: list[BenchmarkCase] = []

    forward_model = SizedMLP(input_size, hidden_size, output_size)

    def forward_trace():
        return forward_model(inputs)

    def validate_forward_trace() -> None:
        result = forward_trace()
        assert result.shape == (batch_size, output_size)
        assert math.isfinite(float(result.data[0, 0]))

    benchmarks.append(BenchmarkCase(
        name="training.mlp_forward_trace/64x32x64x16",
        run=forward_trace,
        validate=validate_forward_trace,
        work_items=forward_work,
        gc_enabled=True,
        description="matrix-heavy MLP forward pass with fresh graph tracing",
        layer="training",
        backends=_ACCELERATED,
    ))

    forward_trace()
    computation = forward_model.computation

    def forward_replay() -> ts.Tensor:
        return computation.forward()

    def validate_forward_replay() -> None:
        result = forward_replay()
        assert result.shape == (batch_size, output_size)
        assert math.isfinite(float(result[0, 0]))

    benchmarks.append(BenchmarkCase(
        name="training.mlp_forward_replay/64x32x64x16",
        run=forward_replay,
        validate=validate_forward_replay,
        work_items=forward_work,
        description="recorded matrix-heavy MLP forward replay",
        layer="training",
        backends=_ACCELERATED,
    ))

    prediction = forward_replay()

    def loss_only() -> ts.Tensor:
        return ts.mean((prediction - targets) ** 2.0)

    def validate_loss_only() -> None:
        result = loss_only()
        assert result.size == 1
        assert math.isfinite(float(result.item()))

    benchmarks.append(BenchmarkCase(
        name="training.mse_loss/64x16",
        run=loss_only,
        validate=validate_loss_only,
        work_items=batch_size * output_size,
        description="MSE loss over a precomputed model prediction",
        layer="training",
        backends=_ACCELERATED,
    ))

    backward_model = SizedMLP(input_size, hidden_size, output_size)
    backward_prediction = backward_model(inputs)
    backward_loss = ts.mean((backward_prediction - targets) ** 2.0)
    backward_optimizer = ts.optim.SGD(
        backward_model.parameters(),
        learning_rate=1e-4,
    )

    def backward_only() -> None:
        backward_optimizer.zero_grad()
        ts.backward(backward_loss)

    def validate_backward_only() -> None:
        backward_only()
        assert all(
            parameter.grad is not None
            for parameter in backward_model.parameters()
        )

    benchmarks.append(BenchmarkCase(
        name="training.mlp_backward/64x32x64x16",
        run=backward_only,
        validate=validate_backward_only,
        work_items=2 * forward_work,
        description="reverse pass through a prebuilt matrix-heavy MLP graph",
        layer="training",
        backends=_ACCELERATED,
    ))

    optimizer_model = SizedMLP(input_size, hidden_size, output_size)
    for parameter in optimizer_model.parameters():
        parameter.grad = ts.full(parameter.shape, 0.01)
    optimizer = ts.optim.SGD(
        optimizer_model.parameters(),
        learning_rate=1e-4,
    )
    parameter_count = sum(
        parameter.size for parameter in optimizer_model.parameters()
    )

    def optimizer_only() -> None:
        optimizer.step()

    def validate_optimizer_only() -> None:
        optimizer_only()
        assert all(
            math.isfinite(float(parameter.data._data[0]))
            for parameter in optimizer_model.parameters()
        )

    benchmarks.append(BenchmarkCase(
        name="training.mlp_optimizer_sgd/32x64x16",
        run=optimizer_only,
        validate=validate_optimizer_only,
        work_items=parameter_count,
        description="SGD phase with precomputed gradients",
        layer="training",
        backends=_ACCELERATED,
    ))

    step_model = SizedMLP(input_size, hidden_size, output_size)
    step_optimizer = ts.optim.SGD(
        step_model.parameters(),
        learning_rate=1e-4,
    )

    def complete_step():
        result = step_model(inputs)
        loss = ts.mean((result - targets) ** 2.0)
        step_optimizer.zero_grad()
        ts.backward(loss)
        step_optimizer.step()
        return loss

    def validate_complete_step() -> None:
        result = complete_step()
        assert result.size == 1
        assert math.isfinite(float(result.data.item()))

    benchmarks.append(BenchmarkCase(
        name="training.mlp_step/64x32x64x16",
        run=complete_step,
        validate=validate_complete_step,
        work_items=3 * forward_work + parameter_count,
        gc_enabled=True,
        description="complete matrix-heavy MLP training step",
        layer="training",
        backends=_ACCELERATED,
    ))
    return benchmarks


def cases() -> list[BenchmarkCase]:
    """Return tiny latency and medium matrix-heavy phase benchmarks."""
    benchmarks = core_cases()
    if ts.get_backend() != "python":
        benchmarks.extend(_medium_cases())
    return benchmarks

"""End-to-end model training-step benchmarks."""

from __future__ import annotations

import math

import tensors as ts

from .runner import BenchmarkCase


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


def cases() -> list[BenchmarkCase]:
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
        )
    ]

"""A fully connected network of configurable depth and width."""

from __future__ import annotations

from typing import Any

import tensors as ts

from ...inputs import tensor


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
            weights.append(
                ts.Variable(
                    tensor(
                        (fan_in, fan_out),
                        dtype_name=dtype_name,
                        kind="constant",
                        value=scale,
                    ),
                    name=f"weight_{index}",
                )
            )
            biases.append(
                ts.Variable(
                    tensor(
                        (fan_out,),
                        dtype_name=dtype_name,
                        kind="constant",
                        value=0.01,
                    ),
                    name=f"bias_{index}",
                )
            )
        # Assigned as tuples so the Graph metaclass finds them as parameters.
        self.weights = tuple(weights)
        self.biases = tuple(biases)
        self.depth = depth

    def forward(self, inputs: Any) -> Any:
        current = inputs
        last = len(self.weights) - 1
        for index, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            current = current @ weight + bias
            if index != last:
                current = ts.relu(current)
        return current


__all__ = ["MultiLayerPerceptron"]

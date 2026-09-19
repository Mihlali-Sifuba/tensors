"""One affine layer, and one model that contains another.

The smallest structures that still record a graph: enough to measure
what building, compiling, and replaying one costs, and what a nested
model adds to each.
"""

from __future__ import annotations

from typing import Any

import tensors as ts

from ...inputs import tensor


class Linear(ts.Graph):
    """A single affine layer, as a structural Graph model."""

    def __init__(self, features: int, units: int) -> None:
        super().__init__()
        self.weight = ts.Variable(
            tensor(
                (features, units), dtype_name="float64", kind="constant", value=0.05
            ),
            name="weight",
        )
        self.bias = ts.Variable(
            tensor((units,), dtype_name="float64", kind="constant", value=0.01),
            name="bias",
        )

    def forward(self, inputs: Any) -> Any:
        return ts.relu(inputs @ self.weight + self.bias)


class Nested(ts.Graph):
    """A model built from two nested Graph models."""

    def __init__(self, features: int, units: int) -> None:
        super().__init__()
        self.first = Linear(features, units)
        self.second = Linear(units, units)

    def forward(self, inputs: Any) -> Any:
        return self.second(self.first(inputs))


__all__ = ["Linear", "Nested"]

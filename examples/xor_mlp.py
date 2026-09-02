"""Train a small XOR MLP using the parameter-initialization API."""

import tensors as ts


class MLP(ts.Graph):
    """Two-layer perceptron: 2 inputs -> 8 ReLU units -> 1 probability."""

    def __init__(self) -> None:
        super().__init__()

        hidden_initializer = ts.init.HeNormal(dtype=ts.float64)
        output_initializer = ts.init.XavierNormal(dtype=ts.float64)

        self.hidden_weight = ts.Variable(
            hidden_initializer((2, 8)),
            name="hidden_weight",
        )
        self.hidden_bias = ts.Variable(
            ts.zeros((8,), dtype=ts.float64),
            name="hidden_bias",
        )
        self.output_weight = ts.Variable(
            output_initializer((8, 1)),
            name="output_weight",
        )
        self.output_bias = ts.Variable(
            ts.zeros((1,), dtype=ts.float64),
            name="output_bias",
        )

    def forward(self, inputs):
        hidden = ts.relu(
            inputs @ self.hidden_weight + self.hidden_bias
        )
        logits = hidden @ self.output_weight + self.output_bias
        return ts.sigmoid(logits)


def describe(name: str, tensor: ts.Tensor) -> None:
    """Print basic sample statistics for an initialized parameter."""
    mean = ts.mean(tensor).item()
    stddev = ts.std(tensor).item()
    print(
        f"{name}: shape={tensor.shape}, dtype={tensor.dtype.name}, "
        f"mean={mean:.4f}, std={stddev:.4f}"
    )


ts.random.seed(42)

feature_rows = [
    [-1.0, -1.0],
    [-1.0, 1.0],
    [1.0, -1.0],
    [1.0, 1.0],
]
target_values = [0.0, 1.0, 1.0, 0.0]
features = ts.Tensor(feature_rows)
targets = ts.Tensor([[value] for value in target_values])

model = MLP()
describe("hidden weight", model.hidden_weight.data)
describe("output weight", model.output_weight.data)

optimizer = ts.optim.Adam(model.parameters(), learning_rate=0.03)

initial_predictions = model(features)
initial_loss = ts.mean((initial_predictions - targets) ** 2.0)
print("initial loss:", initial_loss.data.item())

for step in range(1, 1501):
    predictions = model(features)
    loss = ts.mean((predictions - targets) ** 2.0)

    optimizer.zero_grad()
    ts.backward(loss)
    optimizer.step()

    if step % 300 == 0:
        print(f"step {step:4d} loss: {loss.data.item():.8f}")

final_predictions = model(features)
final_loss = ts.mean((final_predictions - targets) ** 2.0)

print("final loss:", final_loss.data.item())
print("predictions:")
for inputs, target, prediction in zip(
    feature_rows,
    target_values,
    final_predictions.data.tolist(),
):
    print(
        f"  x={inputs} target={target:.0f} "
        f"prediction={prediction:.4f}"
    )

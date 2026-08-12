"""Train a small multilayer perceptron using the public tensors API."""

import tensors as ts


class MLP(ts.Graph):
    """A one-hidden-layer network: input -> linear -> ReLU -> linear."""

    def __init__(self) -> None:
        super().__init__()

        # One input feature, two hidden units, and one output feature.
        self.input_to_hidden = ts.Variable([[0.5, 0.25]], name="W₁")
        self.hidden_bias = ts.Variable([0.1, 0.1], name="b₁")
        self.hidden_to_output = ts.Variable([[0.25], [0.25]], name="W₂")
        self.output_bias = ts.Variable([0.0], name="b₂")

    def forward(self, inputs):
        hidden = ts.relu(inputs @ self.input_to_hidden + self.hidden_bias)
        return hidden @ self.hidden_to_output + self.output_bias


# Learn y = 2x + 1 from four training examples.
features = ts.Tensor([[0.0], [1.0], [2.0], [3.0]])
targets = ts.Tensor([[1.0], [3.0], [5.0], [7.0]])

model = MLP()
optimizer = ts.optim.Adam(model.parameters(), learning_rate=0.05)

for _ in range(300):
    predictions = model(features)
    loss = ts.mean((predictions - targets) ** 2.0)

    optimizer.zero_grad()
    ts.backward(loss)
    optimizer.step()

predictions = model(features)
loss = ts.mean((predictions - targets) ** 2.0)

print("mean squared error:", loss.data.item())
print("predictions:", predictions.data.tolist())
print("learned parameters:")
for parameter in model.parameters():
    print(f"  {parameter.name}:", parameter.data.tolist())

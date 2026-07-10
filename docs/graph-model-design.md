# Graph as a Model Function

## Status

Proposed design. This document describes the intended public API and
responsibilities; it does not describe the current implementation exactly.

## Goal

`Graph` represents a reusable, differentiable computational function while
preserving ordinary, eager Python model code.

```python
class Linear(Graph):
    def __init__(self):
        super().__init__()
        self.weight = Variable([0.1], name="weight")
        self.bias = Variable([0.0], name="bias")

    def forward(self, x):
        return x * self.weight + self.bias
```

Calling the model remains Pythonic:

```python
prediction = model(x)
```

The call eagerly computes a result and also uses the model's persistent graph
representation. A user opts into graph construction by defining or wrapping a
`Graph`; ordinary Python functions and classes remain eager-only.

## Responsibilities

A `Graph` owns the computational representation of its function:

- parameters and child graphs;
- the input bindings used by a call;
- operation nodes and edges;
- the output or outputs produced by `forward`.

A `Graph` does not own training policy:

- targets and datasets;
- loss functions;
- optimisers and learning rates;
- epochs, batches, and training loops.

This gives a clear boundary:

```text
inputs -> model graph -> prediction
prediction + target -> loss -> gradients -> parameter update
```

## Model API

`forward` is model-author code. It defines the function represented by the
graph. `Graph.__call__` is the public execution entry point.

```python
model = Linear()
prediction = model(x)  # calls model.forward(x) through Graph.__call__
```

The existing graph-execution method named `forward` should therefore be
renamed internally, for example to `_replay`, `_run`, or `_execute`.

## Functional Model API

For a small functional model, `Graph` can also be used directly as a Python
decorator:

```python
weight = Variable([0.1], name="weight")
bias = Variable([0.0], name="bias")


@Graph
def model(x):
    return x * weight + bias
```

This is equivalent to:

```python
def model_function(x):
    return x * weight + bias


model = Graph(model_function)
```

After decoration, `model` is a callable graph-function rather than a plain
Python function:

```python
prediction = model(x)
```

`@Graph` is deliberately preferred over a separate `@Graph.function`
decorator. A graph is a function representation, so the direct decorator is
both concise and Python-native. It complements the subclass API:

```python
@Graph
def model(x):
    ...


class Model(Graph):
    def forward(self, x):
        ...
```

Use the functional form for small models whose state can be captured from the
surrounding scope. Use subclassing for models with explicit state, nested
graphs, or more substantial behaviour.

## Execution Lifecycle

The intended lifecycle is:

1. Construct the model and its persistent parameters.
2. On the first call, run `forward` eagerly and record the operations as the
   model graph.
3. On later compatible calls, bind new input values and replay the stored
   graph.
4. When the function's traced structure is no longer compatible with the new
   input shapes or control-flow path, rebuild explicitly.

Illustrative API:

```python
prediction = model(x)  # builds graph on first call; replays it thereafter
model.rebuild(x)       # explicitly trace again when required
```

The exact compatibility and rebuild policy remains a design decision, but it
must be explicit and predictable.

## Training Is External

Training composes a model graph with an eager loss expression. The model does
not receive a loss during construction.

```python
from tensors import Graph, Variable, backward, mean
from tensors.optim import SGD


class Linear(Graph):
    def __init__(self):
        super().__init__()
        self.weight = Variable([0.1], name="weight")
        self.bias = Variable([0.0], name="bias")

    def forward(self, x):
        return x * self.weight + self.bias


def mse(prediction, target):
    error = prediction - target
    return mean(error * error)


model = Linear()
optimizer = SGD(model.parameters(), learning_rate=0.1)

for x_value, target_value in [([1.0], [3.0]), ([2.0], [5.0])]:
    x = Variable(x_value, requires_grad=False)
    target = Variable(target_value, requires_grad=False)

    prediction = model(x)
    loss = mse(prediction, target)

    optimizer.zero_grad()
    backward(loss)
    optimizer.step()
```

`backward(loss)` begins at the loss and propagates through the model output
into the model graph and its parameters. A future `loss.backward()` API would
express the same idea more directly.

## Composition

Graphs compose by being stored as attributes of other graphs and called from
their parent's `forward` method.

```python
class Network(Graph):
    def __init__(self):
        super().__init__()
        self.first = Linear()
        self.second = Linear()

    def forward(self, x):
        return self.second(self.first(x))
```

`Network.parameters()` should recursively return parameters from child graphs,
allowing an optimiser to update the complete model.

## Design Principles

- Model code uses normal Python expressions.
- Eager execution and graph construction coexist in a graph call.
- Graph construction is opt-in through `Graph`, not globally ambient.
- A graph is a function representation, not a training loop.
- Losses and optimisers are composable code outside the model definition.
- The model graph must be inspectable and reusable independently of training.

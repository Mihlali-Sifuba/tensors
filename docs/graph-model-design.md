# Graph as a Model Function

## Status

Current implementation. This document describes the public `Graph` API and
its execution model.

## Goal

`Graph` represents a reusable, differentiable computational function while
preserving ordinary, eager Python model code.

`Graph` is the model abstraction because a mathematical model is a function.
It does not introduce a separate layer or module hierarchy that model authors
must translate their equations into. Persistent trainable values are ordinary
`Variable` attributes, child models are ordinary `Graph` attributes, and
`forward` states the computation with normal Python expressions.

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

Each call eagerly computes a result and records a fresh graph for that
execution. A user opts into reusable model capture by defining or wrapping a
`Graph`. Operations on `Variable` values still record autograd history when
they occur in ordinary Python functions or classes.

## Responsibilities

A `Graph` retains the latest execution metadata for its function:

- persistent parameters and child graphs stored as normal Python attributes;
- the output or outputs produced by the latest `forward` call;
- computations rooted at those outputs;
- the reachable operation nodes and edges.

Latest execution metadata is stored per thread. Parameters and other mutable
model attributes remain shared Python state and are not synchronized by
`Graph`.

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

Tensor inputs are wrapped as non-trainable `Variable` values before `forward`
runs. Existing `Variable` inputs are preserved, and ordinary Python arguments
are passed through unchanged.

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

The lifecycle is:

1. Construct the model and its persistent parameters.
2. On every call, convert Tensor inputs to non-trainable Variables and execute
   `forward` eagerly.
3. Record the operations created by that call and capture the nodes and edges
   reachable from each returned Variable.
4. Store the outputs and computations as the calling thread's latest Graph
   execution metadata.

Calling the model again records a new computation; it does not bind values into
or replay a cached graph. `rebuild(*args, **kwargs)` currently has the same
behavior as a normal call and exists as an explicit retracing entry point.
Previously returned Variables retain their own computation history when kept
by the caller. `release()` only drops the Graph object's references to its
latest execution on the calling thread.

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

`Network.parameters()` recursively returns parameters from child graphs,
allowing an optimiser to update the complete model. Parameter discovery also
traverses common containers and values captured by functional Graph closures.

## Design Principles

- A model is represented as a callable mathematical function.
- Model code uses normal Python expressions.
- Trainable state consists of `Variable` attributes discovered without manual
  parameter registration.
- Eager execution and fresh graph recording coexist in every graph call.
- Reusable model capture is opt-in through `Graph`; eager `Variable`
  operations record their own autograd history independently.
- A graph is a function representation, not a training loop.
- Losses and optimisers are composable code outside the model definition.
- The latest model graph is inspectable, and each captured `Computation` can be
  replayed independently of training while it remains active.
- Public abstractions must clarify the mathematics or enable necessary
  behaviour; they should not add ceremony around expressions Python already
  represents clearly.

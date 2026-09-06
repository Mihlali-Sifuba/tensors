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

## Recorded structure

A traced `Graph` records the same vertex alternation as any eager expression:

```text
VariableNode -> OperationNode -> VariableNode
```

`Graph` owns none of that structure directly. It holds the outputs of its most
recent call and the `Computation` objects rooted at them; the graph itself is
reachable through those outputs. See
[Automatic differentiation](autodiff.md) for the vertex and edge contract.

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

A subclass model is built as it is constructed:

1. `Model()` allocates the instance and runs the subclass `__init__`, which
   creates the parameters.
2. Once `__init__` has returned, `Graph` records the model's graph: vertices
   stand in for its inputs, `forward` runs structurally over them and the
   parameters' own vertices, and nothing is calculated.
3. The recorded graph is compiled into the `Computation` the model owns from
   then on, with its input vertices as the program's boundaries.
4. `model(x)` binds the call's Tensors to those input vertices and replays
   the program, so the Python `forward` body is not run again.

Because one program is replayed, a built model returns the same output
Variable object on every call, holding that call's value, and its graph
identity is stable. `model.nodes`, `model.edges` and `model.computation`
therefore describe the model from construction onwards, before its first
call.

A model keeps the earlier tracing lifecycle when it cannot be described
before its values exist: the functional `@Graph` form, a `forward` taking
configuration arguments whose values are only known per call, and a `forward`
whose expression the graph cannot record yet — a Python scalar operand, or a
function that has no structural form. A `Variable` input also traces, because
its autograd identity belongs to the caller rather than to the model's own
input vertex.

That last case is recognized by one signal and nothing else. Recording an
expression the graph cannot describe raises
`UnsupportedStructuralExpression`, and only that abandons the build. Anything
else — a misspelled attribute, a `forward` that fails, a value read off an
input, a malformed output, a failure inside the compiler — is a real error
and construction reports it where it happened.

Traced calls behave as before: each records a new computation. `compile(*args,
**kwargs)` explicitly enables guarded replay for Tensor inputs on the calling
thread. Matching backend, shape, dtype, keyword layout, and static-argument
guards rebind the recorded input Variables and execute the existing plan.
Guard misses retrace and replace the cached plan. Variable inputs always trace
freshly so their autograd identity is preserved.

`rebuild(*args, **kwargs)` bypasses a matching compiled trace and refreshes it;
`uncompile()` restores ordinary fresh tracing. Compiled calls reuse their
returned Variable objects and update their Tensor values. Previously returned
Variables from fresh traces retain their own computation history when kept by
the caller. `release()` drops both the latest execution and any compiled trace
on the calling thread.

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
Nested calls treat their explicit inputs as local traversal boundaries and
defer child-plan compilation until child metadata is requested. The outer
Graph still records one complete plan, avoiding repeated upstream traversal.

## Design Principles

- A model is represented as a callable mathematical function.
- Model code uses normal Python expressions.
- Trainable state consists of `Variable` attributes discovered without manual
  parameter registration.
- Eager execution and fresh graph recording are the default; guarded replay is
  explicit through `compile()`.
- Reusable model capture is opt-in through `Graph`; eager `Variable`
  operations record their own autograd history independently.
- A graph is a function representation, not a training loop.
- Losses and optimisers are composable code outside the model definition.
- The latest model graph is inspectable, and each captured `Computation` can be
  replayed independently of training while it remains active.
- Public abstractions must clarify the mathematics or enable necessary
  behaviour; they should not add ceremony around expressions Python already
  represents clearly.

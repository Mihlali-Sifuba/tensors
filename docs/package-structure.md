# Package Structure and Namespace

## Status

Proposed design. This document describes the intended package hierarchy and
public import surface; it does not describe the current implementation exactly.

## Public API

The canonical user import is:

```python
import tensors as ts
```

The root package is a small ergonomic facade:

```text
ts.Tensor
ts.Variable
ts.Graph
ts.backward
ts.float32
ts.float64
```

This lets ordinary model code stay concise:

```python
import tensors as ts


class Linear(ts.Graph):
    def __init__(self):
        super().__init__()
        self.weight = ts.Variable([0.1])
        self.bias = ts.Variable([0.0])

    def forward(self, x):
        return x * self.weight + self.bias
```

The organised subpackages remain available for advanced use, but model authors
should not need to import internal operation or graph-node classes.

## Target Folder Layout

```text
tensors/
├── __init__.py            # root public facade
├── tensor.py              # Tensor storage, shapes, dtypes, indexing
├── variable.py            # differentiable value type
├── dtype.py
├── ops/                   # basic algebra only
│   ├── __init__.py
│   ├── add.py
│   ├── sub.py
│   ├── mul.py
│   ├── div.py
│   └── neg.py
├── linalg/                # linear algebra
│   ├── __init__.py
│   ├── dot.py
│   ├── matmul.py
│   └── transpose.py
├── math/                  # mathematical functions
│   ├── __init__.py
│   ├── exp.py
│   ├── log.py
│   ├── sqrt.py
│   ├── sin.py
│   ├── cos.py
│   ├── tanh.py
│   ├── relu.py
│   ├── sigmoid.py
│   ├── sum.py
│   ├── mean.py
│   ├── min.py
│   ├── max.py
│   └── std.py
├── optim/                 # optimisers, added with the training API
│   └── __init__.py
└── graph/                 # reusable graph-functions/models
    ├── __init__.py
    ├── graph.py
    ├── node.py
    └── edge.py
```

## Namespace Rules

The namespace mirrors the package hierarchy:

```python
ts.ops.add(x, y)
ts.linalg.matmul(x, weight)
ts.math.exp(x)
ts.math.mean(x)
```

Common math functions are also root aliases for concise model code:

```python
ts.exp(x)
ts.relu(x)
ts.mean(x)
```

The folders have deliberately narrow responsibilities:

- `ops` contains basic algebra only: addition, subtraction, multiplication,
  division, and negation.
- `linalg` contains linear-algebra operations such as dot products and matrix
  multiplication.
- `math` contains all mathematical functions, including reductions and
  activation functions. It remains flat rather than separating activations or
  reductions into extra namespaces.
- `graph` owns reusable computational model functions and their nodes/edges.
- `Computation` owns forward and backward execution; `ts.backward` is a root
  convenience alias.
- `optim` will own parameter-update algorithms when the training API is built.

Operation classes, `Node`, and `Edge` are implementation or advanced
inspection details. They should not be part of the everyday root API.

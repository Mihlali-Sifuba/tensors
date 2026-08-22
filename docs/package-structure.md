# Package Structure and Namespace

## Status

Current implementation. This document describes the package hierarchy and
public import surface provided by the project.

## API Philosophy

The public API is organised around mathematical expression rather than
framework ceremony. Users should be able to translate a formula into Python
without first translating it into a hierarchy of framework-specific objects:

```python
logits = inputs @ weight + bias
loss = ts.cross_entropy(logits, targets)
```

Python operators express algebra, root-level functions carry familiar
mathematical names, and both `Tensor` and `Variable` participate in the same
expression language. `Graph` represents a reusable mathematical function;
trainable values are normal `Variable` attributes and are discovered without
manual registration.

The package hierarchy exists to organise implementation and support
discoverability. It must not force users to navigate subpackages or import
parallel layer, module, and parameter abstractions merely to state ordinary
mathematics. New public abstractions should make the mathematics clearer or
provide behaviour that cannot be expressed cleanly with the existing
vocabulary.

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

## Folder Layout (Abridged)

```text
tensors/
├── __init__.py            # root public facade
├── _typing.py             # shared public type aliases
├── tensor.py              # Tensor storage, construction, and indexing
├── variable.py            # differentiable value type
├── dtype.py               # dtype definitions and promotion
├── casting.py             # storage conversion helpers
├── creation.py            # tensor-value constructors
├── py.typed               # marker for inline package typing
├── utils/                 # internal shape and broadcasting helpers
│   ├── broadcasting.py
│   ├── indexing.py
│   ├── lists.py
│   ├── shape.py
│   └── slicing.py
├── ops/                   # primitive differentiable operations
│   ├── add.py, sub.py, mul.py, div.py, neg.py
│   └── pow.py, slice.py, cast.py
├── linalg/                # linear algebra
│   ├── dot.py
│   ├── matmul.py
│   ├── norm.py
│   ├── outer.py
│   └── transpose.py
├── math/                  # functions, reductions, activations, and losses
├── optim/                 # parameter-update algorithms
│   ├── optimizer.py
│   ├── sgd.py
│   ├── adam.py
│   └── rmsprop.py
└── graph/                 # tracing, execution, and differentiation
    ├── graph.py           # reusable callable model abstraction
    ├── computation.py     # forward replay and reverse-mode execution
    ├── derivatives.py     # Jacobian and Hessian construction
    ├── gradcheck.py       # finite-difference verification
    ├── node.py
    ├── edge.py
    ├── protocols.py
    └── state.py
```

## Namespace Rules

The public API supports both root conveniences and organised subpackages:

```python
ts.add(x, y)
ts.linalg.matmul(x, weight)
ts.math.exp(x)
ts.math.mean(x)
```

Primitive operation classes remain available through `ts.ops` for advanced
inspection, and the compatibility namespace exposes calls such as
`ts.Ops.add(x, y)`.

Common math functions are also root aliases for concise model code:

```python
ts.exp(x)
ts.relu(x)
ts.mean(x)
```

The folders have deliberately narrow responsibilities:

- `creation` provides public constructors for mathematically defined tensor
  values, including zeros, ones, ranges, and identity matrices.
- `ops` contains primitive differentiable operations such as arithmetic,
  powers, slicing, and casting.
- `utils` contains internal shape, row-major indexing, and broadcasting helpers.
  It is not re-exported from the root package.
- `linalg` contains linear-algebra operations such as dot products and matrix
  multiplication.
- `math` contains all mathematical functions, including reductions and
  activation functions. It remains flat rather than separating activations or
  reductions into extra namespaces.
- `graph` owns tracing state, operation protocols, computation execution,
  derivatives, and reusable computational model functions.
- `Computation` owns forward and backward execution; `ts.backward` is a root
  convenience alias.
- `optim` provides the shared optimizer contract plus SGD, Adam, and RMSprop.

Operation classes, `Node`, and `Edge` are implementation or advanced
inspection details. They should not be part of the everyday root API.

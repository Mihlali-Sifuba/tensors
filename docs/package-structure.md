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
ts.Shape
ts.Strides
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
├── backend/               # backend selection and optional kernels
│   ├── __init__.py
│   ├── _array.py          # shared NumPy/CuPy kernel implementation
│   ├── numpy.py
│   └── cuda.py
├── storage/               # internal native storage implementations
│   ├── __init__.py
│   ├── _base.py
│   ├── _conversion.py
│   ├── python.py
│   ├── numpy.py
│   └── cuda.py
├── _typing.py             # shared public type aliases
├── shape.py               # immutable logical tensor extents
├── strides.py             # immutable physical storage movement
├── tensor.py              # Tensor storage, construction, and indexing
├── variable.py            # differentiable value type
├── dtype.py               # dtype definitions and promotion
├── casting.py             # storage conversion helpers
├── creation.py            # tensor-value constructors
├── py.typed               # marker for inline package typing
├── utils/                 # indexing, slicing, and Tensor broadcasting
│   ├── broadcasting.py
│   ├── coordinates.py
│   ├── indexing.py
│   ├── lists.py
│   └── slicing.py
├── ops/                   # the Operation contract and its implementations
│   ├── operation.py       # the Operation abstract base class
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
├── init/                  # functional parameter initialization
│   ├── initializer.py     # callable initializer base contract
│   ├── variance_scaling.py
│   ├── xavier_uniform.py, xavier_normal.py
│   ├── he_uniform.py, he_normal.py
│   ├── lecun_uniform.py, lecun_normal.py
│   └── truncated_normal.py, orthogonal.py
├── random/                # backend-native random generation
│   └── _state.py
└── graph/                 # tracing, execution, and differentiation
    ├── graph.py           # reusable callable model abstraction
    ├── computation.py     # executable computation planning and execution
    ├── autograd.py        # functional reverse-mode differentiation API
    ├── fusion.py          # fused execution of compatible instruction runs
    ├── derivatives.py     # Jacobian and Hessian construction
    ├── gradcheck.py       # finite-difference verification
    ├── node.py            # Node, VariableNode, and OperationNode
    ├── edge.py
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

- `backend` owns process and context-local selection, cached internal kernel
  dispatch, provider-neutral array kernels, and optional NumPy/CUDA entry
  points.
- `storage` owns the internal Python, NumPy, and CUDA representations and their
  lazy conversion cache. Storage classes are not a second public tensor API.
- `Shape` owns logical dimensions, rank, size, tuple-like slicing of its
  dimension values (for example, `Shape(2, 3, 4)[1:]`), and pure
  broadcast-shape inference. `Strides` owns physical traversal metadata and
  canonical contiguous-stride construction. Both are root-level immutable
  value objects returned by `Tensor.shape` and `Tensor.strides`. Ordinary
  construction derives them automatically; see
  [Tensor memory model](memory-model.md).
- `creation` provides public constructors for mathematically defined tensor
  values, including zeros, ones, ranges, and identity matrices.
- `ops` owns `Operation`, the abstract contract every concrete mathematical
  operation implements, plus the primitive differentiable operations such as
  arithmetic, powers, slicing, and casting. The graph package references an
  operation through `OperationNode` and `Instruction`; it does not define what
  an operation is.
- `utils/coordinates.py` converts between logical coordinates and canonical
  row-major logical linear indices using `Shape` and canonical contiguous
  strides derived from that `Shape`; arbitrary Tensor strides and `offset`
  do not participate.
  `utils/indexing.py` normalizes Tensor indices and combines `Shape`,
  `Strides`, and `offset` to produce physical storage indices.
  `utils/slicing.py` owns Tensor/Python indexing and slicing semantics (for
  example, `tensor[1:, :, 2]`), while `utils/broadcasting.py` materializes
  Tensor broadcasting. Pure broadcast-shape inference lives on `Shape`, while
  stride construction lives on `Strides`. The `utils` package is not
  re-exported from the root.
- `linalg` contains linear-algebra operations such as dot products and matrix
  multiplication.
- `math` contains all mathematical functions, including reductions and
  activation functions. It remains flat rather than separating activations or
  reductions into extra namespaces.
- `graph` owns tracing state, the operation contract, computation execution,
  derivatives, and reusable computational model functions. A recorded graph
  alternates `VariableNode -> OperationNode -> VariableNode`, and every
  relationship is an `Edge`. `Node` holds only identity and connectivity;
  `VariableNode` adds its `Variable` and `OperationNode` adds its `Operation`.
  An operation defines how a local derivative is calculated; `Computation`
  decides which local derivatives a reverse pass requires and supplies that
  demand as `needs_input_grad`. See [Automatic differentiation](autodiff.md) for
  the recorded topology and the responsibility split.
- `Computation` owns the instruction plan and executes it forwards and in
  reverse; `fusion` recognizes and accelerates compatible instruction runs
  beside that plan; `autograd` is the functional interface through which
  callers request differentiation. `ts.backward` is a root convenience alias.
- `optim` provides the shared optimizer contract plus SGD, Adam, and RMSprop.
- `init` provides immutable callable initializer configurations plus lowercase
  function facades for variance-scaling, Xavier, He, LeCun, truncated-normal,
  and orthogonal initialization. Both forms remain under `ts.init` instead of
  expanding the root facade.
- `random` owns seeded Python, NumPy, and CUDA generator state and exposes the
  minimal `ts.random` facade. It does not modify provider-global RNG state.

Operation classes, `Node`, `VariableNode`, `OperationNode`, and `Edge` are
implementation or advanced inspection details. They should not be part of the
everyday root API.

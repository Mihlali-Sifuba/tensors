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
├── backend/               # backend selection, dispatch, and kernels
│   ├── __init__.py        # backend facade and re-exports
│   ├── types.py           # backend and operation type aliases
│   ├── config.py          # selection, availability, and configuration
│   ├── policy.py          # workload-size policy for acceleration
│   ├── loading.py         # lazy loading of a backend's kernel package
│   ├── storage.py         # the Storage contract every backend implements
│   ├── conversion.py      # conversion between backend representations
│   ├── dispatch/          # one execute_* entry point per operation
│   │   ├── __init__.py    # dispatch facade
│   │   ├── arithmetic/    # add.py, subtract.py, multiply.py, ...
│   │   ├── elementwise/   # exp.py, exp_gradient.py, clip.py, where.py, ...
│   │   ├── creation/      # arange.py, eye.py, full.py, linspace.py, ...
│   │   ├── manipulation/  # cast.py, concat.py, slice.py, transpose.py, ...
│   │   ├── reductions/    # reduce_sum.py, reduce_sum_gradient.py, ...
│   │   ├── linalg/        # matmul.py, matmul_gradient.py, outer.py, ...
│   │   ├── convolution/   # convolution.py, convolution_gradient.py
│   │   ├── fusion/        # fused_elementwise.py and its VJP
│   │   ├── nn/            # softmax.py, cross_entropy.py, ...
│   │   └── optim/         # sgd_update.py, adam_update.py, ...
│   ├── python/            # the reference backend
│   │   ├── storage.py     # array.array storage
│   │   └── kernels/       # one module per operation, by domain
│   │       ├── arithmetic/    # add.py, subtract.py, ...
│   │       ├── elementwise/   # exp.py, exp_gradient.py, ...
│   │       ├── creation/
│   │       ├── manipulation/
│   │       ├── reductions/
│   │       ├── linalg/
│   │       ├── convolution/
│   │       ├── fusion/
│   │       ├── nn/            # plus _normalization.py, shared axis helpers
│   │       └── optim/
│   ├── numpy/             # the NumPy backend
│   │   ├── storage.py     # numpy.ndarray storage
│   │   ├── conversion.py  # the Tensor/Storage to ndarray boundary
│   │   └── kernels/       # the same domains, implemented with NumPy
│   └── cuda/              # the CUDA backend
│       ├── storage.py     # device-resident cupy.ndarray storage
│       ├── conversion.py  # the Tensor/Storage to device-array boundary
│       └── kernels/       # the same domains, implemented with CuPy
│           ├── fusion/        # generated-kernel compilation and execution
│           │   ├── expressions.py  # step, operand, derivative expressions
│           │   ├── source.py       # CUDA source assembly
│           │   ├── errors.py       # domain guards and error reporting
│           │   ├── common.py       # operand marshalling for both passes
│           │   ├── fused_elementwise.py
│           │   └── fused_elementwise_backward.py
│           ├── convolution/   # common.py holds padding, tiling, columns
│           └── optim/         # batch_kernels.py holds the fused updates
├── _typing.py             # shared public type aliases
├── shape.py               # immutable logical tensor extents
├── strides.py             # immutable physical storage movement
├── tensor.py              # Tensor storage, construction, and indexing
├── variable.py            # differentiable value type; eager operations
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
├── operations/            # the semantic operation hierarchy (canonical)
│   ├── base.py            # the Operation abstract base class, UNARY_DEMAND
│   ├── _gradient_shaping.py  # cross-domain VJP shaping operations
│   ├── arithmetic/        # add.py, subtract.py, multiply.py, divide.py,
│   │                      # negate.py, power.py
│   ├── elementary/        # abs.py, sign.py, sqrt.py, exp.py, log.py
│   ├── trigonometric/     # sin.py, cos.py, tan.py, arcsin.py, arccos.py,
│   │                      # arctan.py
│   ├── hyperbolic/        # sinh.py, cosh.py, tanh.py, arcsinh.py,
│   │                      # arccosh.py, arctanh.py
│   ├── activations/       # relu.py, sigmoid.py, softplus.py
│   ├── comparison/        # equal.py, not_equal.py, less.py, less_equal.py,
│   │                      # greater.py, greater_equal.py, _operands.py
│   ├── selection/         # minimum.py, maximum.py, clip.py, where.py,
│   │                      # _extremum.py
│   ├── reductions/        # sum.py, mean.py, prod.py, min.py, max.py,
│   │                      # variance.py, std.py, norm.py, logsumexp.py,
│   │                      # argmin.py, argmax.py, _arg_extremum.py
│   ├── manipulation/      # reshape.py, transpose.py, concat.py, stack.py,
│   │                      # slice.py, cast.py
│   ├── linalg/            # matmul.py, outer.py
│   ├── normalization/     # softmax.py, log_softmax.py
│   ├── losses/            # binary_cross_entropy.py, cross_entropy.py
│   └── convolution/       # convolution.py
├── ops/                   # facade: the Operation contract and primitives
├── linalg/                # facade: linear algebra
├── math/                  # facade: functions, reductions, activations, losses
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
└── graph/                 # structural graph representation and tracing
    ├── graph.py           # reusable callable model abstraction
    ├── node.py            # Node, VariableNode, and OperationNode
    ├── edge.py
    ├── expression.py     # applying an operation: structural or runtime
    ├── state.py          # tracing registry; records operation topology
    └── computation/       # the executable, differentiable form of a graph
        ├── instruction.py   # one executable operation invocation
        ├── compiler.py      # Node/Edge topology to slots, instructions, views
        ├── computation.py   # execution of the compiled slot representation
        ├── autograd.py      # functional reverse-mode differentiation API
        ├── gradients.py     # seed construction, accumulation, VJP validation
        ├── derivatives.py   # Jacobian and Hessian construction
        ├── gradcheck.py     # finite-difference verification
        └── fusion.py        # fused execution of compatible instruction runs
```

## Namespace Rules

The public API supports both root conveniences and organised subpackages:

```python
ts.add(x, y)
ts.linalg.matmul(x, weight)
ts.math.exp(x)
ts.math.mean(x)
```

`tensors.operations` is where every operation is defined, grouped by what it
means. `ts.math`, `ts.linalg`, and `ts.ops` are convenience namespaces over
it: they re-export selected names and define nothing of their own, so
`ts.math.exp` and `tensors.operations.elementary.exp` are the same object.
Import from `tensors.operations` when the semantic domain matters, and from a
facade when the shorter name reads better.

```python
from tensors.operations.linalg import matmul
from tensors.operations.reductions import mean
from tensors.ops import Operation      # defined in tensors.operations.base
```

The `Ops` static-method namespace has been removed. The functions it wrapped
are root-level calls:

```python
ts.add(x, y)        # was ts.Ops.add(x, y)
ts.subtract(x, y)
ts.multiply(x, y)
ts.divide(x, y)
```

Common math functions are also root aliases for concise model code:

```python
ts.exp(x)
ts.relu(x)
ts.mean(x)
```

The folders have deliberately narrow responsibilities:

- `backend` owns process and context-local selection, per-operation dispatch,
  and one kernel package per backend. Its package module is a facade: `types`
  names the backends and their operations, `config` selects one and reports
  availability, `policy` decides when a workload is worth accelerating,
  `loading` resolves a kernel package for the selected backend, `storage`
  states the contract every backend's storage implements, and `conversion`
  moves values between those representations.
- `backend.dispatch` holds one module per operation, grouped into domain
  packages. A dispatch module reads the policy, calls the selected backend's
  kernel, and runs the Python reference itself when either declines, so an
  `execute_*` function returns a result rather than a decision. It never
  imports a sibling dispatch module. The two exceptions that may still return
  `None` are optional optimizations whose alternative is ordinary execution
  rather than a reference kernel: `fusion` and the batched `optim` updates.
- `backend.python`, `backend.numpy`, and `backend.cuda` each own one backend
  end to end: its `storage`, its array boundary (`conversion`, for the two
  array backends), and a `kernels` package holding one module per operation
  under the same domain names dispatch uses. The three implementations are
  deliberately independent — NumPy and CUDA do not share a numerical body, and
  neither imports the other — so "where is this operation implemented for this
  backend" has one answer per backend. A backend's `kernels/__init__.py`
  re-exports every kernel under the flat name the loader resolves, which is
  the only name dispatch knows.
- Within a backend, a module is shared only where several operations genuinely
  need the same helper: the Python backend's `nn/_normalization.py` holds the
  axis traversal and softmax components its softmax-family kernels share, and
  the CUDA backend's `fusion` and `convolution` packages hold the source
  assembly, domain guards, and tiling their kernels build on. These never
  import the operation modules above them.
- The Python backend is the reference implementation: it defines shape, dtype,
  error, and differentiation semantics, and it evaluates them in ordinary
  Python. It never calls a public operation class to produce its own result.
- No kernels package exposes a submodule and an exported kernel under the same
  name. Where a module would collide with a Python builtin or the kernel it
  defines, the module and kernel take a qualified name together, so
  `manipulation/cast_tensor.py` defines `cast_tensor` for every backend.
- `Shape` owns logical dimensions, rank, size, tuple-like slicing of its
  dimension values (for example, `Shape(2, 3, 4)[1:]`), and pure
  broadcast-shape inference. `Strides` owns physical traversal metadata and
  canonical contiguous-stride construction. Both are root-level immutable
  value objects returned by `Tensor.shape` and `Tensor.strides`. Ordinary
  construction derives them automatically; see
  [Tensor memory model](memory-model.md).
- `creation` provides public constructors for mathematically defined tensor
  values, including zeros, ones, ranges, and identity matrices.
- `operations` owns every computation the library exposes, grouped by what
  the operation *means* rather than by how a backend executes it or by which
  Python syntax reaches it. `base.py` holds `Operation`, the abstract contract
  every concrete operation implements, and `UNARY_DEMAND`. The graph package
  references an operation through `OperationNode` and `Instruction`; it does
  not define what an operation is.

  An operation module owns its input and configuration validation, its output
  shape and dtype, the dispatch call that evaluates it, the storage it wraps,
  and its reverse-mode and higher-order derivative rules. It never names a
  concrete backend: choosing one is dispatch's job.

  A semantic domain and a backend execution domain are different
  classifications and need not share names. `trigonometric`, `hyperbolic`,
  `activations`, and `selection` all dispatch to backend *elementwise*
  kernels; the semantic folder says what the operation means, the backend
  folder says how it runs.

  Two modules start with an underscore because they are shared structure
  rather than operations: `_gradient_shaping.py` holds the small graph
  operations a VJP needs to broadcast a gradient back to an operand's shape,
  and each `_extremum.py`, `_arg_extremum.py`, and `_operands.py` holds what a
  pair or family of neighbouring operations genuinely shares.
- `ops`, `linalg`, and `math` are public facade packages. Each is a single
  `__init__.py` that re-exports names from `tensors.operations`; none of them
  holds an implementation. They exist because they are documented, familiar
  user-facing namespaces, not as compatibility adapters.
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
- **Broadcasting is one responsibility, and it is not computation.**
  `Shape.broadcast_with` agrees a common shape between two operands;
  `broadcast_to(tensor, shape)` expands one operand to it; `broadcast_tensors`
  does both for a pair. None of them performs arithmetic, comparison or
  selection, resolves a result dtype, or selects a backend. An operation
  broadcasts its operands and then computes for itself, so the broadcasting
  utility never needs to know which operation consumes its result.

  `broadcast_binary_values`, which took two tensors *and an operation* and
  applied it while walking broadcast offsets, is gone. It mixed the two
  responsibilities: every consumer had to hand its arithmetic to the
  broadcasting layer, and the broadcasting layer could not be reused or
  reasoned about without one. The Python reference kernels — the only callers
  it had, since NumPy and CuPy broadcast natively — now read

  ```python
  left = broadcast_to(left, output_shape)._data
  right = broadcast_to(right, output_shape)._data
  values = [evaluate(x, y) for x, y in zip(left, right)]
  ```

  **This costs performance on the Python backend**, by roughly four times on a
  broadcasting binary operation. The single-pass helper allocated only its
  output; two `broadcast_to` calls materialize two full intermediate tensors
  first. Equal-shape operations are unaffected, because `broadcast_to` returns
  its argument unchanged when the shape already matches. Recovering the
  difference means broadcasting without materializing — zero-stride views —
  which needs the storage-ownership design recorded in
  [the memory model](memory-model.md), and is deliberately not attempted here.
- `matmul` is one operation with two public names. `ts.matmul` and `ts.dot`
  both contract the final axes with matrix-product semantics and broadcast any
  leading batch axes; `dot` is an entry point to the same `MatMul` class, not
  a second contraction. The class is named for what it does, and matches the
  vocabulary dispatch and the kernels already use.
- `graph` owns the recorded structure, tracing state, and the public `Graph`
  abstraction; its `computation` subpackage owns the executable,
  differentiable form of that structure. A recorded graph
  alternates `VariableNode -> OperationNode -> VariableNode`, and every
  relationship is an `Edge`. `Node` holds only identity and connectivity;
  `VariableNode` is the graph identity of one value and binds the `Variable`
  materializing it, which may happen after the vertex is recorded, and
  `OperationNode` adds its `Operation`. `GraphState.record_operation` is
  where one invocation's vertices and ordered edges are assembled, so a
  graph can record an operation without anything executing it, and
  `expression` decides which application an expression asked for: runtime
  operands calculate now, and a vertex operand records structure only.
  An operation defines how a local derivative is calculated; `Computation`
  decides which local derivatives a reverse pass requires and supplies that
  demand as `needs_input_grad`. See [Automatic differentiation](autodiff.md) for
  the recorded topology and the responsibility split.
- `graph/computation` holds that executable form. `Compiler` is the boundary
  between the two domains, and one compilation feeds both sides:

  ```text
             structural graph
                    │
                    ▼
                 Compiler
                    │
        ┌───────────┴───────────┐
        ▼                       ▼
  Graph metadata          execution metadata
  (nodes, edges)                │
                                ▼
                           Computation
                                │
                         forward / backward
  ```

  `Compiler` is the last component that understands Nodes and Edges: it
  takes the output vertices to compile, numbers a slot per `VariableNode`,
  emits the slot-based program, resolves each output's execution view, and
  hands the graph layer the traversal and edges it keeps. Compilation reads
  no values, so a graph compiles before the values it names exist. An
  `Instruction` is one executable operation invocation. `Computation`
  receives an already-resolved execution view and works only in the compiled
  domain — vertices, slots, values, instructions, and fusion metadata — to
  execute it forwards and in reverse. It holds Tensors in slots while a pass
  runs and gives each result to the vertex naming its slot, materializing
  that vertex's Variable on the first pass and updating it on later ones.
  `gradients` supplies the generic mechanics reverse execution uses along the
  way — upstream seed construction, gradient accumulation, and VJP result
  validation; `fusion` recognizes and accelerates compatible instruction runs
  beside the program; `autograd` is the functional interface through which
  callers request differentiation; `derivatives` builds Jacobians and Hessians
  from repeated reverse passes and `gradcheck` verifies them against finite
  differences. The subpackage layout is internal: `ts.backward`, `ts.grad`,
  `ts.jacobian`, `ts.hessian`, `ts.gradcheck`, and `ts.graph.Computation` are
  unaffected by it.
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

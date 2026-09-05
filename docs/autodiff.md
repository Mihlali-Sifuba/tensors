# Automatic differentiation

The package records eager `Variable` operations as a directed acyclic graph and
uses reverse-mode automatic differentiation to calculate vector-Jacobian
products (VJPs).

## Graph model

A recorded graph alternates between two vertex types:

```text
VariableNode -> OperationNode -> VariableNode
```

Every relationship is an `Edge`. An operation's operands arrive on its incoming
edges, and its single result leaves on its outgoing edge. For `c = a + b`:

```text
VariableNode(a) ──input_0──┐
                           ▼
                    OperationNode(Add())
                           ▲
VariableNode(b) ──input_1──┘
                           │
                        result
                           ▼
                    VariableNode(c)
```

Responsibilities divide as follows:

| Object | Responsibility |
| --- | --- |
| `Variable` | the differentiable runtime value |
| `VariableNode` | the graph representation of one `Variable` |
| `Operation` | one concrete mathematical invocation (owned by `ts.ops`) |
| `OperationNode` | the graph representation of that invocation |
| `Edge` | a graph relationship and its data flow |
| `Computation` | traversal and execution of the graph |
| `GraphState` | non-owning registry of live nodes and edges |
| `Graph` | the reusable callable function or model |

Stated compactly:

```text
Operation   = local mathematical semantics
Computation = traversal and execution
Graph       = reusable function/model abstraction
```

The same split governs differentiation:

```text
Operation
    defines how local derivatives are calculated

Computation
    determines which local derivatives are required
```

`Node` itself carries only identity and connectivity. `VariableNode` adds its
`variable`, and `OperationNode` adds its `operation`; neither stores execution
state.

### Variables and their nodes

Every `Variable` owns exactly one `VariableNode`, and the relationship is
strong in both directions:

```python
variable.node.variable is variable  # always true
```

This holds for leaves, for Tensor operands wrapped on the way into an
operation, for normalized scalar operands, and for operation results.
`Variable.node` is never an `OperationNode`.

### Operands are graph values, configuration is not

A runtime operand always enters an operation through the graph. Writing
`y = x + 3` records the scalar as a non-gradient `Variable`:

```text
VariableNode(x) ──input_0──┐
                           ▼
                    OperationNode(Add())
                           ▲
VariableNode(3) ──input_1──┘
```

Operand order carries the meaning of a reverse expression, so `3 / x` records
the numerator as `input_0` and `x` as `input_1`. There is no scalar or reverse
flag on the node. Converting a Python scalar preserves the existing dtype
promotion rules, so `int32_variable * 3` still produces `int32`.

Configuration that defines the transformation rather than supplying a value
belongs to the operation instance:

```text
VariableNode(x)
      │
      ▼
OperationNode(Sum(axis=1, keepdims=True))
      │
      ▼
VariableNode(result)
```

`axis`, `keepdims`, a cast dtype, a slice key, and convolution geometry are
configuration. They never appear as graph operands.

### Operation instances

An `Operation` is immutable once constructed, so a recorded invocation cannot
change meaning while a graph still refers to it:

```python
operation = Sum(axis=1)
operation.axis = 0  # AttributeError
```

Operations must not store values produced by a particular forward pass — no
saved inputs, outputs, temporary gradients, or workspaces. That state belongs
to `Computation`, which keeps it per thread so replay and concurrent execution
stay correct.

Reverse demand is execution state for the same reason. An operation never
records which of its operands will be differentiated: that depends on the
reverse call being made, not on the recorded graph. Configuration therefore
never contains a name like `differentiate_left` or `needs_input_grad`, and
`requires_grad` never appears in an operation's VJP logic to decide whether a
derivative is worth calculating.

Each invocation produces exactly one output. A `Graph` returning several
Variables exposes them as separate computation roots.

### Reverse gradient demand

`Computation` resolves demand before executing any VJP and passes it to the
operation as `needs_input_grad`, one flag per operand:

```python
def backward(self, gradient, *inputs, needs_input_grad):
    need_left, need_right = needs_input_grad
    return (
        left_vjp(...) if need_left else None,
        right_vjp(...) if need_right else None,
    )
```

The two absent-versus-zero cases are distinct and enforced:

```text
None
    the VJP was not requested

a zero Tensor/Variable
    the VJP was requested and its mathematical value is zero
```

Returning a value for an unrequested operand raises, as does returning `None`
for a requested one. That makes skipping unused work an enforceable contract
rather than an optimisation an operation may quietly ignore. `backward_graph`
follows the same contract with `Variable` results.

A derivative-specific domain error is raised only when the derivative it
guards was requested. Differentiating `base ** exponent` with respect to a
negative base is well defined, so:

```python
base = ts.Variable([-2.0])
exponent = ts.Variable([2.0])
output = base ** exponent

ts.grad(output, base)      # -4.0
ts.grad(output, exponent)  # ValueError: requires non-negative bases
```

`backward()` requests a gradient at every reachable differentiable Variable,
which is what it publishes. `grad(output, inputs)` instead plans the reverse
pass from the requested inputs: a VJP runs only where a requested Variable's
influence flows toward the output. For

```python
y = (a * b) + c
ts.grad(y, a)
```

the addition is asked for its left VJP only, the multiplication for `a` only,
and nothing behind `c` is calculated. Requesting `c` alone skips the
multiplication entirely. Higher-order differentiation, `create_graph=True`,
`jacobian`, and `hessian` all use the same plan.

Because demand depends on `requires_grad`, which is mutable and participates
in mutation detection, it is resolved per reverse invocation and never cached
on a compiled forward instruction. A replayed computation therefore uses the
gradient requirements that hold at the time it is differentiated.

## Public API

```python
import tensors as ts

x = ts.Variable([2.0, 3.0])
loss = ts.sum(x ** 2.0)

ts.backward(loss)
print(x.grad.tolist())  # [4.0, 6.0]

gradient = ts.grad(loss, x)
print(gradient.tolist())  # [4.0, 6.0]
```

`ts.backward(output)` stores gradients on every reachable trainable
`Variable`. Each call clears gradients from that computation before starting,
so repeated calls do not accidentally accumulate stale values.

`ts.grad(output, inputs)` returns only the requested gradients. It returns
`None` for an input that is not connected to `output`. It is functional:
calling it never clears, replaces, or accumulates any `Variable.grad` value.
This deliberately differs from `ts.backward`, whose job is to populate
`.grad` fields for optimisation.

For a non-scalar output, omitting the upstream gradient differentiates the sum
of its elements. Pass `grad=` to `backward` or `grad_outputs=` to `grad` to
calculate a different VJP. The upstream gradient shape must exactly match the
output shape.

Set `create_graph=True` when a returned gradient will itself be differentiated:

```python
x = ts.Variable([3.0])
y = x ** 3.0

first = ts.grad(y, x, create_graph=True)
second = ts.grad(first, x)
```

## Jacobians and Hessians

`ts.jacobian(output, inputs)` returns every first derivative rather than the
single vector-Jacobian product returned by `ts.grad`. For each input, its result
shape is exactly `output.shape + input.shape`:

```python
x = ts.Variable([2.0, 3.0])
y = ts.concat([x[0] ** 2.0, x[0] * x[1]])

jacobian = ts.jacobian(y, x)
print(jacobian.tolist())  # [4.0, 0.0, 3.0, 2.0]
print(jacobian.shape)     # (2, 2)
```

`ts.hessian(output, inputs)` returns every second derivative. The output must
contain exactly one element. For one input, the result shape is
`input.shape + input.shape`:

```python
x = ts.Variable([2.0, 3.0])
y = ts.sum(x[0] ** 2.0 + x[0] * x[1])

hessian = ts.hessian(y, x)
print(hessian.tolist())  # [2.0, 1.0, 1.0, 0.0]
print(hessian.shape)     # (2, 2)
```

Passing multiple inputs to `jacobian` returns one result per input. Passing
multiple inputs to `hessian` returns a tuple of tuples of Hessian blocks. A
block at `[i][j]` has shape `inputs[i].shape + inputs[j].shape`.

Disconnected inputs produce explicit zero Jacobians or Hessian blocks. Both
functions preserve existing `.grad` attributes. Set `create_graph=True` to
keep the returned derivative matrix differentiable:

```python
x = ts.Variable([2.0])
y = ts.sum(x ** 3.0)

hessian = ts.hessian(y, x, create_graph=True)
third = ts.grad(ts.sum(hessian), x)
```

Constructing a complete Jacobian requires one reverse pass per output element.
Constructing a complete Hessian can therefore be expensive for large tensors;
use `ts.grad(..., grad_outputs=vector)` when only a vector product is needed.

## Numerical verification

`ts.gradcheck` compares the analytical gradient with central finite
differences. It checks the sum of every output element and leaves the caller's
eager graph state unchanged.

```python
inputs = ts.Tensor([[0.2, -0.4, 0.7]])

ts.gradcheck(
    lambda x: ts.softmax(x, axis=1) ** 2.0,
    inputs,
)
```

Float64 inputs are recommended. Avoid testing exactly at discontinuities or
nondifferentiable points. `GradcheckError` reports the input and element where
a mismatch occurs. Pass `raise_exception=False` to receive `False` instead.

## Supported differentiation

First-order gradients are implemented and numerically checked for:

- broadcast arithmetic and powers;
- `sqrt`, `exp`, `log`, `sin`, `cos`, `tan`, `arcsin`, `arccos`, `arctan`,
  `sinh`, `cosh`, `tanh`, `arcsinh`, `arccosh`, `arctanh`, `sign`, `relu`,
  `sigmoid`, and `softplus`;
- axis-aware `sum`, `mean`, `variance`, `min`, `max`, `std`, `norm`, `softmax`,
  `logsumexp`, and `log_softmax`;
- stable multiclass and binary cross-entropy losses;
- `dot`, `matmul`, `outer`, and `transpose`;
- `conv1d`, `conv2d`, and `conv3d`, including input, kernel, and bias gradients;
- `reshape`, slicing, `concat`, and `stack`.

Higher-order gradients are supported for the smooth operations above,
including broadcasting, batched matrix multiplication, softmax, slicing,
concatenation, and stacking. The engine raises `NotImplementedError` when an
operation does not provide a differentiable VJP instead of silently detaching
the gradient.

## Mathematical boundaries

Production behavior includes explicit domain rules:

- `log(x)` requires `x > 0`.
- `tan(x)` is undefined at odd multiples of pi/2; its value and derivative
  grow without bound when floating-point inputs approach those poles.
- `arcsin(x)` and `arccos(x)` require `-1 <= x <= 1`. Their values exist at
  the endpoints, but their finite real derivatives are undefined there.
- `arctan(x)` is defined and differentiable for every finite real input.
- `sinh(x)`, `cosh(x)`, and `arcsinh(x)` are defined and differentiable for
  every real input. Very large `sinh` and `cosh` values overflow to signed or
  positive infinity, respectively.
- `arccosh(x)` requires `x >= 1`. Its value exists at `x == 1`, but its finite
  real derivative is undefined there.
- `arctanh(x)` requires `-1 < x < 1`.
- `sign(x)` has a zero derivative away from zero and raises when differentiated
  at zero, where the function is discontinuous.
- `sqrt(x)` accepts `x >= 0`, but its derivative raises at `x == 0` because the
  finite real derivative is undefined there.
- A differentiable tensor exponent in `base ** exponent` requires a positive
  base. A constant integer-valued exponent can differentiate negative bases.
- `norm` and `std` use a zero first-order subgradient at a zero-magnitude
  reduction group. Their higher derivatives raise there.
- `variance` is the population variance. It remains smooth at zero variance,
  where both its value and first derivative are zero.
- `relu` uses the conventional zero subgradient at zero.
- `min` and `max` divide the first-order gradient equally among tied extrema.
  Higher-order derivatives are not provided because selection changes are
  nondifferentiable.
- `conv1d`, `conv2d`, and `conv3d` are cross-correlations: the kernel is not
  reversed. Their input, kernel, and bias VJPs remain differentiable when
  `create_graph=True`.
- Higher-order `dot`/`matmul` supports vector-vector, matrix-vector,
  vector-matrix, batched matrix products, and broadcast batch dimensions.
- Higher-order differentiation of empty `mean` and `std` reductions is not
  implemented.

These boundaries are part of the API. Returning an explicit error is safer
than manufacturing a gradient at a point where the mathematics does not define
one.

## Mutation and recomputation

An eager operation calculates its output immediately, and its result Variable
records the state of the operation's operands and of the result itself. Every successful item assignment increments the
tensor's read-only `version` counter. Replacing `Variable.data` is tracked
separately, including replacements made by optimizers.

Differentiation rejects a computation if any recorded value has subsequently
been replaced or modified:

```python
x = ts.Variable([2.0])
y = x ** 2.0
x.data[0] = 3.0

ts.backward(y)  # RuntimeError: an input changed after the forward pass
```

This prevents a derivative from combining stale intermediate values with new
inputs. Run the mathematical expression again to construct a fresh graph, as
normal training loops already do. To deliberately reuse an existing topology,
call `Computation(output).forward()` first; it recomputes every operation and
refreshes the recorded mutation state.

Shape, rank, and dtype metadata are read-only. The backing `_data` member is an
internal implementation detail; writing it directly bypasses the public safety
contract.

## Graph lifetime and traversal

Graph state is a non-owning, thread-local registry used for eager inspection.
It stores callback-free weak references, while nodes keep lightweight weak
references to outgoing edges. A `Variable` and its `VariableNode` refer to each
other strongly, so an unreachable computation forms a reference cycle; ordinary
cycle collection still reclaims it. Active `Graph` traces derive their structure
directly from output-owned incoming edges and skip the redundant registry.
Consequently, a persistent leaf such as a model parameter does not keep every
discarded forward result alive. A live output still owns its incoming edges and
therefore retains everything required for recomputation and differentiation.

`GraphState.clear()` forgets its current registrations without invalidating
any live output. `Graph.release()` drops a reusable Graph object's references
to its most recent outputs and computations; retained output Variables remain
valid, and calling the Graph again records a fresh computation. A `Graph`
keeps this latest execution metadata per thread, so concurrent calls on the
same graph do not overwrite one another's `computation`, `nodes`, or `edges`.
Parameters and other mutable attributes are still shared Python state and
must be synchronized separately if callers modify them concurrently.

`Computation` compiles its dependency-first traversal into ordered
`Instruction` objects once at construction. Each instruction names the
operation to run, the slots holding its operands, and the slot receiving its
result, resolved from the operation vertex's edges at that point, so replay
and differentiation never walk the graph again. `forward` traverses those
instructions and reverse execution traverses them backwards.

Every pass allocates its own value and gradient buffers, so concurrent replays
of one Computation share no mutable execution state.

Fusion is recorded beside the instruction sequence rather than inside it: the
plan maps the first index of each fusible run to where the run ends and what
the backend kernel needs. An instruction absent from that mapping simply
executes on its own, so the same instruction sequence is valid on every
backend, with or without fusion.

The execution plan is an optimized runtime representation and deliberately does
not mirror the graph object for object: the graph is the semantic structure,
and the plan is how that structure is executed.

On CUDA, compatible single-consumer float32 and float64 elementwise chains may
execute as one fused kernel in both directions. Fusion supports broadcast
tensor arithmetic, powers, and common unary mathematics.

Forward fusion depends only on the forward mathematics, the dtype and layout,
and what the backend kernel supports; it never depends on which derivatives a
later reverse pass may want. Backward fusion does consult the current demand:
it produces only the external operand gradients that were requested, and when
a requested derivative is one the compact fused form cannot express, that
group falls back to ordinary operation VJP execution instead of disabling the
valid fused forward pass. The internal chain derivative that carries the
upstream gradient through the sequence is calculated regardless, because
reverse propagation needs it even when it is never published. Intermediate
`.data` and `.grad` values are still published, so fusion changes execution
cost rather than graph semantics.
Native backend VJPs are also used for supported reductions and elementwise
operations; numerically delicate inputs return to the stable Python rules.

Call `Computation.release()` when a long-lived Computation object no longer
needs its output or plan. A released Computation cannot be reused.

## The operation contract

`ts.ops.Operation` is an abstract base class. It lives in the operations
subsystem because it is the contract every concrete mathematical operation
implements; the graph package references an operation rather than defining
what one is. A concrete operation inherits from it and implements `forward()`
and `backward()`:

```python
from tensors.ops import Operation


class Identity(Operation):
    name = "identity"

    def forward(self, value):
        return value

    def backward(self, gradient, value):
        return [gradient]
```

`backward_graph()` is optional and enables higher-order differentiation. The
base implementation raises `NotImplementedError`, so an operation without it
reports the limitation instead of silently detaching a gradient.

A configured operation declares its configuration in `__slots__` and assigns it
in `__init__`, which keeps the instance immutable:

```python
class Scale(Operation):
    __slots__ = ("factor",)
    name = "scale"

    def __init__(self, *, factor):
        object.__setattr__(self, "factor", factor)

    def forward(self, value):
        return value * self.factor

    def backward(self, gradient, value):
        return [gradient * self.factor]
```

`Computation` invokes `operation.forward(...)`, `operation.backward(...)`, and
`operation.backward_graph(...)` directly; it never interprets an operation's
configuration.

## Numerically stable probability functions

Use `logsumexp` and `log_softmax` instead of directly composing `log`, `sum`,
and `exp` when inputs may have large magnitudes:

```python
logits = ts.Variable([[1000.0, -1000.0]])
log_probabilities = ts.log_softmax(logits, axis=1)
```

`cross_entropy` accepts class indices with the class axis removed or dense
target distributions. `binary_cross_entropy` accepts probabilities by default;
pass `from_logits=True` for the stable raw-logit formulation. Both losses
support `reduction="none"`, `"mean"`, and `"sum"`:

```python
labels = ts.Tensor([0], dtype=ts.int64)
loss = ts.cross_entropy(logits, labels)

binary_loss = ts.binary_cross_entropy(
    ts.Variable([1000.0, -1000.0]),
    ts.Tensor([1.0, 0.0]),
    from_logits=True,
)
```

The stable formulations subtract the reduction maximum or use the equivalent
softplus identity, avoiding overflow for large finite logits. Targets for
binary cross-entropy must lie in the closed interval `[0, 1]`.

## Validation guarantees

During backward execution, every operation must return exactly one gradient per
requested input, and `None` for every unrequested one. The engine validates
each gradient's presence, type, and shape before propagating it. Graph traversal is
iterative, so deeply composed functions do not depend on Python's recursion
limit. Node labels are derived from the recorded operation and never control
execution.

Run the full suite from the repository root with:

```powershell
python -m unittest discover -s tests -t .
```

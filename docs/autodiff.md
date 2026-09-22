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
| `VariableNode` | the graph identity of one value it may predate |
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

`Node` itself carries only identity and connectivity. `VariableNode` adds the
value it names, and `OperationNode` adds its `operation`; neither stores
execution state.

### Variables and their nodes

A `VariableNode` is the graph identity of a value. It can exist before that
value has been calculated, which is the order execution works in:

```text
construct graph -> compile -> Computation executes Instructions
    -> Tensor result -> Variable materialized against its VariableNode
```

A leaf already holds its value, so constructing a `Variable` records a vertex
already bound to it. A value the graph only names is recorded unbound and
materialized once execution produces it:

```python
node = VariableNode()                   # the graph names a value
node.is_bound                           # False
node.variable                           # UnboundVariableNodeError

result = node.materialize(tensor, "c")  # execution produced the value
node.variable is result                 # True
result.node is node                     # True
```

Binding is one-time and symmetric, so a vertex names at most one `Variable`
and that `Variable` names it back:

```python
variable.node.variable is variable  # true from materialization onwards
```

This holds for leaves, for Tensor operands wrapped on the way into an
operation, for normalized scalar operands, and for operation results.
`Variable.node` is never an `OperationNode`. Rebinding a vertex, or binding a
`Variable` that already has one, raises rather than silently replacing the
relationship.

Reading structure never requires a materialized value: `producer`,
`operand_nodes`, and `result_node` describe edges. `operands` and `result`
resolve the Variables those vertices name, and raise
`UnboundVariableNodeError` while one is still pending. Compilation is
structural for the same reason: an execution slot is numbered by the vertex
naming a value, so a graph compiles whether or not that value exists.
Executing the compiled program is what still needs one.

### Eager operations execute through the graph

An eager expression is not a shortcut past the graph; it is the graph, run one
operation at a time:

```text
c = a + b
  -> normalize the operands into Variables
  -> GraphState records the invocation: VariableNode(c), OperationNode(Add),
     and the edges ordering the operands and carrying the result
  -> Compiler numbers that fragment's slots
  -> Computation executes it and calls Add.forward(a.data, b.data)
  -> the result Tensor materializes c against VariableNode(c)
```

`GraphState.record_operation` is where that structure is assembled, so the
graph can record an operation without anything executing it.
`Variable._apply_operation` is the single path every operator and every
`math` and `linalg` function takes, and it only orders the layers. None of
them calls `Operation.forward` itself, and the result Variable is
materialized by the Computation rather than constructed around a value that
was calculated first.

The operands of a new operation already hold their values, so they are the
boundaries of its compiled fragment. Compiling `d = c * b` emits one
instruction over `c` and `b`, and the `a + b` behind `c` is not re-executed,
so a chain of `n` eager operations costs `n` fragments rather than `n`
growing replays. Boundaries only limit that forward program: the structural
graph still records every operation, so differentiating a later result
compiles the complete history behind it.

### Structural expressions

The same operators also apply to a `VariableNode`, and there they describe a
graph instead of calculating one:

```python
x = VariableNode()          # a value the graph names but nothing holds
h = x @ weight              # weight is an ordinary Variable parameter
y = relu(h + bias)

y.is_bound                  # False, and no kernel has run
```

The operands decide which application happens. Every operand being a runtime
Variable makes the expression a calculation; a single `VariableNode` operand
makes the whole expression structural, because a value that does not exist
yet cannot take part in one that runs now. A Variable in a structural
expression takes part as the vertex it was materialized against, so a
parameter's value is never read while a graph is being described.

A Tensor operand becomes a non-gradient leaf, since it is a value that
already exists. A Python scalar is rejected: an eager scalar is typed by
promotion against the value beside it, and a structural expression has not
calculated that value, so recording one would mean inventing a dtype. Pass a
typed `Tensor` or `Variable` instead.

That rejection, and a function that has no structural form yet, raise
`UnsupportedStructuralExpression` — a `TypeError` that states a limit of what
the graph can describe rather than a faulty expression. It is the signal a
model build watches for when deciding that a model must be traced instead.

The recorded structure is an ordinary graph, so it compiles like any other:

```python
Compiler((y,), boundaries=(x,)).compile()   # dot, add, relu
```

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
rather than an optimisation an operation may quietly ignore. The contract is
the same when the reverse pass is building a derivative graph; only the kind
of the results changes, from `Tensor` to `Variable`.

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
  at zero, where the function is discontinuous. Its VJP is **routed**: the
  result is canonical `+0.0` whatever the upstream gradient is, so a negative
  upstream does not make it `-0.0` and an infinite or NaN upstream does not
  make it NaN. A NaN primal gives a NaN derivative.
  ([sign-semantics.md](sign-semantics.md) section 6.)
- `abs(x)` uses a zero subgradient at the kink, and its VJP is likewise
  routed rather than multiplied, so the kink is canonical `+0.0` for any
  upstream. The second derivative is zero away from the kink and raises at
  it, because the subgradient there is a choice rather than a limit.
  ([abs-semantics.md](abs-semantics.md) sections 6 and 8.)
- `sqrt(x)` accepts `x >= 0`, but its derivative raises at `x == 0` because the
  finite real derivative is undefined there. A **negative** primal is not an
  error in the VJP: it gives NaN, matching the forward rule. The VJP's
  evaluation order is specified — root, then doubling, then division, each
  rounding in the declared dtype.
  ([sqrt-semantics.md](sqrt-semantics.md) section 6.)
- A differentiable tensor exponent in `base ** exponent` requires a positive
  base. A constant integer-valued exponent can differentiate negative bases.
- `norm` and `std` use a zero first-order subgradient at a zero-magnitude
  reduction group. Their higher derivatives raise there.
- `variance` is the population variance. It remains smooth at zero variance,
  where both its value and first derivative are zero.
- `relu` uses the conventional zero subgradient at zero. Its VJP is routed
  rather than multiplied, so an infinite or NaN upstream on the inactive
  side gives canonical `+0.0` rather than NaN. The second derivative is zero
  away from the kink and raises at it.
  ([relu-semantics.md](relu-semantics.md) sections 6 and 8.)
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

Four of them — `sign`, `abs`, `sqrt` and `relu` — now have their VJPs
specified in their own semantics documents, which govern the first-order
VJP, the graph-built VJP and the named higher-order regions. Two properties
those documents share are worth stating once here, because they were
previously inconsistent between backends:

- **A VJP that discards its upstream gradient routes rather than
  multiplies.** Multiplying by a materialised zero lets a negative upstream
  leave `-0.0` and an infinite or NaN upstream leave NaN on a branch the
  derivative says contributes nothing. Selecting a literal zero does not.
- **The VJP is recorded rather than frozen.** None of the four reads host
  values to build a mask, so a compiled graph replayed with values that
  change branch answers for the values it is replayed with, and a domain
  error that the replayed values deserve is still raised.

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
result. A slot is numbered by the `VariableNode` naming its value, and the
compiler resolves the operand and result slots from the operation vertex's
edges, so replay and differentiation never walk the graph again.

A slot holds a Tensor while a pass runs. `forward` seeds the leaf slots from
the Variables they read, executes each instruction into its output slot, and
gives that slot's vertex the value it produced: the first pass materializes
the Variable the vertex named, and a later pass updates the one already bound
to it. A program compiled from vertices that hold nothing yet therefore runs
exactly like a replay of a recorded one.

Differentiation works on Variables rather than slot values, so it requires a
forward pass to have produced them; a reverse pass over a program whose slots
are still empty says so instead of differentiating an incomplete one. Reverse
execution then traverses the same instructions backwards.

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

### Backend execution during differentiation

Differentiation is execution. A reverse pass runs kernels exactly as a forward
pass does, so it is governed by the same rules, and
[Numerical backends](backends.md) is the authoritative source for them. They
are not restated here.

> **Verified current behaviour, outside the arithmetic contract.** Native
> backend VJPs are used for supported reductions and elementwise operations,
> and numerically delicate inputs return to the stable Python rules. A reverse
> pass for those operations may therefore execute on the Python backend even
> when NumPy or CUDA was selected.

> **Approved target contract, implemented for `+`, `-`, `*` and `/`.** Under
> **explicit** backend selection, a supported operation's VJP must execute on
> the selected backend. If it cannot execute conformingly there, it must raise
> a clear unsupported-operation error. It must not silently run its VJP
> through the Python backend. Under **automatic** selection the resolved
> backend is treated the same way, because `"auto"` resolves once and then
> names a backend like any other selection.
>
> The four arithmetic VJPs meet this at every size. Their computations are
> `sum_to_shape` for `+` and `-`, a negation for the right operand of `-`,
> `sum_products_to_shape` for `*`, and ordinary division for `/`; each
> dispatches through an entry point that consults no workload policy and
> raises `BackendOperationUnsupportedError` rather than answering with another
> backend's kernel.
>
> A VJP outside those four still does not. Operations such as `power`,
> `where`, the losses and the extrema reduce their broadcast gradients through
> `execute_sum_to_shape`, which keeps the workload threshold and the reference
> fallback. Closing that means extending the execution requirement past
> arithmetic, which is separate work, and is why two entry points exist for
> the same reduction.
>
> **Power is closed for its first derivative.** Its two gradient kernels
> dispatch strictly — no threshold, no fallback, no operand-reading decline —
> under [arithmetic semantics G6](arithmetic-semantics.md#1271-rules), and the
> broadcast reduction that shapes their results now asks for the same
> selected-backend execution the `+` and `-` reductions do, so a small
> broadcast power backward pass no longer returns `PythonStorage` where the
> same pass without broadcasting returns `NumPyStorage`. Section 12.7.4 does
> not impose `**`'s own accuracy bounds on a whole gradient expression, and
> the reduction it now uses agrees with the one it replaced to well under an
> ulp, in both directions, on the cases measured.
>
> **Its second derivative is not closed.** `PowerBaseGradient` and
> `PowerExponentGradient` still carry two implementations of their own
> derivative with different guarantees — a range-safe host loop that raises at
> a zero base, and a formula built from ordinary operations that neither
> guards the range nor raises — so differentiating a recorded power gradient
> a second time with `create_graph` has no single answer to give yet.
> Numerical second derivatives are unaffected and use the host loop.

> **Consequence worth knowing.** The array `sum_to_shape` and
> `sum_products_to_shape` kernels decline when a gradient contains an infinity
> or a NaN, and `sum_to_shape` also declines on subnormals, because their
> scaled accumulation cannot carry those values. A decline used to mean a
> quiet trip to the Python reference; for the arithmetic VJPs it now means an
> error. So a reverse pass through `+`, `-` or `*` whose upstream gradient has
> already become non-finite raises under NumPy or CUDA, where it previously
> returned a non-finite gradient computed in Python. Verified on both
> backends. Diverging training is the obvious way to meet this. Teaching those
> kernels to handle non-finite operands natively would remove it without
> weakening the contract, and is not part of this change.

> **Exponentiation.** The derivatives of `**` are specified in
> [arithmetic semantics section 12.7](arithmetic-semantics.md#127-differentiation-d7).
> Three requirements bear on this document. **All three are implemented (D7);
> the descriptions below of what happens "today" are historical.**
>
> - **The two gradients are independent.** A base gradient that exists is
>   returned even when the exponent gradient does not. `(-2.0) ** 3.0` yields a
>   base gradient of `12.0` and an exponent gradient of `NaN`. Before D7 the
>   whole backward pass raised and both were lost.
> - **Differentiation does not raise on a numerical condition**, and no backend
>   may synchronise with the host to detect one. An undefined derivative is
>   `NaN`; an approved one-sided infinite slope is `±inf`, and that convention
>   is confined to the single region that names it.
> - **Gradients execute on the selected backend** at every size, carrying each
>   operand's own declared dtype and reduced to that operand's shape.
>
> The fused CUDA backward used to raise `"power derivative is undefined at a
> zero base"` (error code 14), which would have made the fused and unfused
> passes disagree — the same failure already corrected for division by zero.
> **Code 14 was removed with D7**, together with power's forward codes 8 and 9,
> and the generated kernel now carries the whole region table instead.

Two kinds of fallback are easy to confuse, and only one of them is a backend
fallback:

| Fallback | Stays on the selected backend? | Permitted under explicit selection? |
| --- | --- | --- |
| **Plan-level** — a fused run executes as ordinary unfused operations | yes | yes |
| **Backend-level** — an operation executes on a different backend | no | no; it must raise instead |

Plan-level fallback is the case described above, where a requested derivative
is one the compact fused form cannot express and the group reverts to ordinary
operation VJP execution. That changes the execution plan, not the executor,
and it remains available under explicit selection — provided it satisfies the
equivalence requirement below.

Backend-level fallback is what explicit selection forbids. A reverse pass that
quietly reaches the Python kernel is indistinguishable, from the caller's side,
from one that ran where they asked.

### Numerical equivalence under optimisation

> **Status: implemented for the CUDA fusion kernels.**

An optimisation must preserve the **specified** result of the original
sequence of typed operations. Fusion may remove intermediate allocations and
memory traffic; it may not change what the function computes.

What "preserve" requires depends on what the specification determines, and the
distinction matters now that exponentiation is specified:

| operation | requirement on a fused or relocated execution |
| --- | --- |
| `+`, `-`, `*`, `/` | **Bitwise identical.** IEEE 754 requires these to be correctly rounded, so the result is uniquely determined. Unchanged by this section. |
| integer `**` | **Bitwise identical.** Exact fixed-width arithmetic ([Arithmetic semantics §12.4.1](arithmetic-semantics.md#1241-non-negative-exponents-wrap-d3)). |
| floating `**` | The accuracy contract in [Arithmetic semantics §12.6](arithmetic-semantics.md#126-accuracy): IEEE special values exactly, and every other result within the stated bound of the correctly rounded value. **Bitwise equality is not required and must not be inferred.** |

Whatever the operation, fusion must preserve **the mathematical operation
performed, the declared dtype, the exceptional-value semantics including the
signs of zeros and infinities, and every rounding boundary the unfused form
has**. A fused kernel must not introduce an additional numerical operation,
omit one, or reassociate operations that were not authorised to be
reassociated.

Concretely, for an expression evaluated in dtype `d`, the unfused form rounds
at every operation:

```text
t = round_d(a * b)
r = round_d(t + c)
```

A fused kernel must preserve **both** rounding boundaries. It must not
contract the pair into a fused multiply-add that rounds once, because that
changes the result. On CUDA this is not hypothetical: NVRTC contracts `a*b+c`
into an FMA by default, and compiling the statement shape the fusion kernels
emit, without the per-step stores, makes 25.6% of 65536 random `float64`
triples differ from the two-rounding result.

The fusion kernels are therefore compiled with `--fmad=false`. Measured on
this toolchain the generated source does not currently contract even without
it, because every step is written to memory and that makes each intermediate
observable to the compiler. The option is set so the guarantee follows from
the compilation rather than from a code-generation detail that no test pins
down, and it is not claimed to have changed any result.

A second equivalence failure was real. The fused forward kernel tested every
denominator and raised `ZeroDivisionError`, so an expression returning an
infinity eagerly raised once compiled. Floating division delivers the IEEE
result in both forms now.

The same constraint forbids a fused kernel from carrying an intermediate at
wider precision than the declared dtype, for the reason given in
[Arithmetic semantics §5.5](arithmetic-semantics.md#55-why-declared-precision-matters).

Two consequences follow. Both are stated **per operation**, against the table
above, so that neither claims bitwise equality where the specification does not
provide it:

- **Replay.** A Computation replayed with fusion enabled must produce what the
  same Computation produces without it: every operation in the replayed graph
  performs the same mathematical operation, in the same declared dtype, at the
  same rounding boundaries, and satisfies its own numerical specification on
  the inputs it actually receives — bit for bit where the table requires it,
  and with the same exceptional values. Fusion is an execution plan, and a plan
  does not change the function.
- **Backend switching.** The same graph replayed on a different backend must
  likewise satisfy every operation's own specification: identical results for
  the four arithmetic operations and for integer exponentiation, and for
  floating exponentiation a result satisfying
  [§12.6](arithmetic-semantics.md#126-accuracy) with identical special values,
  signs and dtype. A fused CUDA kernel and an unfused Python execution of one
  expression are two implementations of one specified function; where that
  function is not uniquely determined by the standard, they must both be
  conforming implementations of it rather than bit-for-bit copies of each
  other.

**Conformance is per operation and does not compose into a graph-level bound.**
The limits in
[§12.6.2](arithmetic-semantics.md#1262-the-accuracy-bounds) apply to each
individual power, evaluated on the inputs that power actually receives. They
say nothing about the final value of a computation that contains one. In

```python
t = x ** y
r = t - c
```

two conforming backends may compute different `t`, each within the bound. The
subtraction then meets its own specification exactly, on the inputs it was
given, and `r` still differs. **That difference is permitted, and later
operations may amplify it. A graph containing floating-point power has no
universal graph-level ULP bound under this specification.**

This concerns accuracy alone. It does not license a fused kernel to reassociate
operations, to contract them, or to omit one: the requirements above on the
operation sequence, the declared dtypes and the rounding boundaries are
unchanged, as are the bitwise requirements for the correctly rounded arithmetic
operations and for integer exponentiation.

If a relaxed numerical mode permitting contraction or reassociation is wanted
later, it must be an explicit, separately documented execution mode. It is not
introduced now.

The arithmetic rules themselves — rounding, overflow, dtype, promotion — are
specified once, in [Arithmetic semantics](arithmetic-semantics.md), and are
not restated here.

Call `Computation.release()` when a long-lived Computation object no longer
needs its output or plan. A released Computation cannot be reused.

## The operation contract

`Operation` is an abstract base class defined in
`tensors.operations.base`, and re-exported as `ts.ops.Operation`. It lives in
the operations subsystem because it is the contract every concrete
mathematical operation implements; the graph package references an operation
rather than defining what one is. A concrete operation inherits from it and
implements `forward()` and `backward()`:

```python
from tensors.operations import Operation      # or: from tensors.ops import Operation


class Identity(Operation):
    name = "identity"

    def forward(self, value):
        return value

    def backward(self, gradient, value):
        return [gradient]
```

`backward()` is the whole derivative contract. It used to be joined by
`backward_graph()`, which a reverse pass building a derivative graph called
instead; nothing selects between them now. An operation written against
operations rather than against Tensors serves both passes and can be
differentiated again, and one written against Tensors answers the numerical
pass and stops there.

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

`Computation` invokes `operation.forward(...)` and `operation.backward(...)`
directly; it never interprets an operation's configuration.

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

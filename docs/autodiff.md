# Automatic differentiation

The package records eager `Variable` operations as a directed acyclic graph and
uses reverse-mode automatic differentiation to calculate vector-Jacobian
products (VJPs).

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
  `sign`, `relu`, `sigmoid`, `tanh`, and `softplus`;
- axis-aware `sum`, `mean`, `variance`, `min`, `max`, `std`, `norm`, `softmax`,
  `logsumexp`, and `log_softmax`;
- stable multiclass and binary cross-entropy losses;
- `dot`, `matmul`, `outer`, and `transpose`;
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
- Higher-order `dot`/`matmul` currently requires both operands to have at least
  two dimensions. First-order gradients support vector operands.
- Higher-order differentiation of empty `mean` and `std` reductions is not
  implemented.

These boundaries are part of the API. Returning an explicit error is safer
than manufacturing a gradient at a point where the mathematics does not define
one.

## Mutation and recomputation

An eager operation calculates its output immediately and records the state of
its input and output tensors. Every successful item assignment increments the
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

Graph state is a non-owning, thread-local registry used for tracing and
inspection. It stores weak references, and operation inputs store weak
references to their outgoing edges. Consequently, a persistent leaf such as a
model parameter does not keep every discarded forward result alive. A live
output still owns its incoming edges and therefore retains everything required
for recomputation and differentiation.

`GraphState.clear()` forgets its current registrations without invalidating
any live output. `Graph.release()` drops a reusable Graph object's references
to its most recent outputs and computations; retained output Variables remain
valid, and calling the Graph again records a fresh computation. A `Graph`
keeps this latest execution metadata per thread, so concurrent calls on the
same graph do not overwrite one another's `computation`, `nodes`, or `edges`.
Parameters and other mutable attributes are still shared Python state and
must be synchronized separately if callers modify them concurrently.

`Computation` calculates its dependency-first node order once at construction
and reuses the immutable cached order for forward and backward passes. Call
`Computation.release()` when a long-lived Computation object no longer needs
its output or traversal. A released Computation cannot be reused.

## Operation protocols

Custom operation classes are checked structurally through Python protocols.
`ts.graph.Operation` requires `forward()` and `backward()`;
`HigherOrderOperation` additionally provides `backward_graph()`, while
`ReverseOperation` provides `forward_reverse()` for scalar-left expressions.
The class does not need to inherit from these protocols:

```python
class Identity:
    @staticmethod
    def forward(value):
        return value

    @staticmethod
    def backward(gradient, value):
        return [gradient]
```

Supplying `Identity` as an operation class works because it implements the
required interface. See Python's
[`typing.Protocol`](https://docs.python.org/3.14/library/typing.html#typing.Protocol)
documentation for structural subtyping details.

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
input. The engine validates each gradient's type and shape before propagating
it. Graph traversal is iterative, so deeply composed functions do not depend on
Python's recursion limit. Node labels are descriptive metadata and never
control execution.

Run the full suite from the repository root with:

```powershell
python -m unittest discover -s . -p "test*.py" -v
```

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
`None` for an input that is not connected to `output` and clears any old
gradient on that input.

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
- `sqrt`, `exp`, `log`, `relu`, `sigmoid`, `tanh`, and `softplus`;
- axis-aware `sum`, `mean`, `min`, `max`, `std`, `norm`, and `softmax`;
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
- `sqrt(x)` accepts `x >= 0`, but its derivative raises at `x == 0` because the
  finite real derivative is undefined there.
- A differentiable tensor exponent in `base ** exponent` requires a positive
  base. A constant integer-valued exponent can differentiate negative bases.
- `norm` and `std` use a zero first-order subgradient at a zero-magnitude
  reduction group. Their higher derivatives raise there.
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

An eager operation calculates its output immediately. If leaf `.data` values
are replaced later, call `Computation(output).forward()` before differentiation
to refresh intermediate values. Do not mutate leaves between a forward pass and
its backward pass unless the computation is deliberately recomputed.

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

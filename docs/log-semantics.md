# Natural-logarithm semantics

This document defines the forward and first-order reverse-mode contract for
`ts.log`. Higher-order differentiation and `create_graph` are outside this
milestone.

## 1. Forward contract

`log` is the elementwise natural logarithm

\[
F(x) = \log(x).
\]

The output shape is the input shape. A `float32` input produces `float32` and
a `float64` input produces `float64`. Every integer dtype is converted to
`float64` before evaluation and produces `float64`. The conversion happens
before the domain rule is applied numerically by the kernel.

For positive finite inputs each backend converts the operand to binary64,
evaluates its native binary64 logarithm, then rounds once into the declared
output dtype. This promises the correctly narrowed provider result, not a
platform-independent correctly rounded transcendental for every real input.

The domain and exceptional-value table is:

| input | result |
| --- | --- |
| positive finite | the narrowed logarithm described above |
| `+inf` | `+inf` |
| NaN | NaN; payload and sign are unspecified |
| `+0.0` or `-0.0` | raises `ValueError` |
| negative finite or `-inf` | raises `ValueError` |

The domain exception applies to the whole tensor: one non-positive element
prevents a result. NaN is not classified as non-positive and propagates.
An empty tensor contains no invalid element and returns an empty result.

Positive subnormal operands are valid. They must not be confused with zero.
NumPy and Python preserve them directly. CUDA widens binary32 through the
shared PTX conversion before its domain reduction and before evaluation,
because the ordinary device conversion and comparison flush subnormals on
this toolchain.

## 2. Domain checking and synchronisation

The dispatcher owns the public domain check because it owns the prepared
native operand. Python uses `any`, NumPy uses a native reduction, and CUDA uses
a device reduction on the PTX-widened values. CUDA then transfers one boolean
to the host so `ValueError` can be raised immediately. That synchronisation
is a deliberate consequence of the public exception contract; the tensor is
never materialised on the host.

The numerical kernels do not inspect values or repeat the domain check.

## 3. Execution contract

Explicit `python`, `numpy`, or `cuda` selection is an execution requirement:

- every tensor size executes on the selected backend;
- the dispatcher validates operand residency, lowers to native values, calls
  that backend's kernel, and validates result residency;
- there is no workload threshold or Python-reference fallback;
- a backend that declines raises `BackendOperationUnsupportedError`.

`Log.forward` owns output shape and dtype. `execute_log` owns selection,
residency, lowering, the domain reduction, and invocation. Kernels receive
native values and perform only the logarithm.

## 4. First-order VJP

For upstream gradient `G`,

\[
G\frac{\partial F}{\partial x} = \frac{G}{x}.
\]

This is evaluated as one binary64 division and narrowed once to the primal
dtype. It is not evaluated as `(1 / x) * G`, which would introduce a second
rounding.

The VJP does not repeat the forward domain check. A primal reached through a
successful public forward is positive or NaN, and repeating the reduction
would add unnecessary work and a CUDA synchronisation to every reverse pass.
A directly invoked VJP follows the floating division rules in
`arithmetic-semantics.md` section 7.2, including infinities and NaNs at zero.

The upstream gradient must exactly match the primal's shape and dtype. The
operation validates both before dispatch. If `needs_input_grad[0]` is false,
the operation returns `None` without validation or numerical work.

## 5. Fusion

CUDA fusion widens stored binary32 values through PTX before evaluating the
logarithm and its domain predicate. It therefore accepts positive subnormals,
raises for both signed zeros and negative values, and narrows the result under
the same format rule as eager execution. Tests compare both routes to the
written contract, not merely to one another.
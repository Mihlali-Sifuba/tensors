# Exponential semantics

This document defines the forward and first-order reverse-mode contract for
`ts.exp`. Higher-order differentiation and `create_graph` are outside this
milestone.

## 1. Forward contract

`exp` is the elementwise function

\[
F(x) = e^x.
\]

The output shape is the input shape. A `float32` input produces `float32` and
a `float64` input produces `float64`. Every integer dtype is converted to
`float64` before evaluation and produces `float64`. That conversion is a
format conversion, not an exact-integer promise: an `int64` beyond `2**53`
may round before its exponential is evaluated.

For finite inputs each backend converts the operand to binary64, evaluates
its native binary64 exponential, then rounds once into the declared output
dtype. Thus a `float32` result is not produced by binary32 intermediate
arithmetic. This contract promises the correctly narrowed provider result; it
does not claim that every platform transcendental library is correctly
rounded for every real input.

The exceptional-value table is:

| input or condition | result |
| --- | --- |
| `+0.0` or `-0.0` | exactly `1.0` |
| `+inf` | `+inf` |
| `-inf` | `+0.0` |
| NaN | NaN; payload and sign are unspecified |
| result above the finite range | `+inf` |
| result below the smallest subnormal | `+0.0` |
| representable subnormal result | that subnormal result; it is not flushed |

Overflow is therefore a value, not an error. Python's `math.exp` raises
`OverflowError`, so the Python kernel translates that exception to `+inf`.
NumPy and CUDA narrow the binary64 result directly; they do not route it
through the older declining storage conversion, which rejected a legitimate
binary32 infinity.

CUDA uses the shared PTX `_widen` and `_narrow` conversions around binary32.
Ordinary device casts flush subnormals on this toolchain, including results in
the exponential's underflow band. The PTX conversions preserve gradual
underflow.

## 2. Execution contract

Explicit `python`, `numpy`, or `cuda` selection is an execution requirement:

- every tensor size executes on the selected backend;
- the dispatcher validates operand residency, lowers to native values, calls
  that backend's kernel, and validates result residency;
- there is no workload threshold or Python-reference fallback;
- a backend that declines raises `BackendOperationUnsupportedError`.

`Exp.forward` owns output shape and dtype. `execute_exp` owns selection,
residency, lowering, and invocation. Kernels receive native values and perform
only the numerical calculation.

## 3. First-order VJP

For upstream gradient `G`,

\[
G\frac{\partial F}{\partial x} = G e^x.
\]

The primal and upstream are read in their declared dtype, widened to binary64,
the exponential and multiplication are evaluated there, and the result is
narrowed once to the primal dtype. IEEE multiplication governs exceptional
values: for example `1 * exp(+inf)` is `+inf`, while
`0 * exp(+inf)` is NaN.

The upstream gradient must exactly match the primal's shape and dtype. The
operation validates both before dispatch. If `needs_input_grad[0]` is false,
the operation returns `None` without validation or numerical work.

## 4. Fusion

CUDA fusion evaluates the expression in binary64 and uses the same PTX
narrowing for `float32`, so eager and fused results follow this contract.
Conformance tests compare each route to mathematical expectations rather than
using one backend as the oracle.
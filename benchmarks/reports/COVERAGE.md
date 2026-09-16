# Coverage decisions, and what cannot be meaningfully benchmarked

Every gap below is a decision with a reason, not an omission. Gaps that the
harness itself records as `unsupported` appear in the merged report with the
same reasons attached to each affected case.

## Dtypes

**Swept:** `float64`, `float32` everywhere; `int64`, `int32` for binary
arithmetic, comparisons, and creation.

**Not swept, and why:**

- `int16`, `int8`, `uint8`. The kernels branch on `dtype.kind == "integer"`,
  not on width, so these take the identical path to `int64`/`int32`. They
  differ only in overflow limits, and the benchmark's ramp values
  (1…97) overflow `int8` and `uint8` under multiplication — which would
  measure the overflow fallback rather than the arithmetic. Their behavior is
  therefore represented by the `int64`/`int32` measurements.
- **There is no boolean dtype.** `tensors.dtype` defines exactly `float64`,
  `float32`, `int64`, `int32`, `int16`, `int8`, `uint8`. Comparisons
  (`equal`, `less`, `greater_equal`, …) return `uint8`, and `where` consumes
  `uint8` as a mask. "Bool where relevant" is therefore covered by the
  comparison and selection benchmarks, whose results are `uint8` tensors;
  there is no separate boolean path to measure.

## Tensor layouts

**Measured:** contiguous; contiguous at a non-zero storage offset; transposed;
strided; broadcast (zero-stride); and the conversion of each to a compact
tensor.

**Cannot be measured through the public API:** view creation. No public
operation returns a view — `reshape`, `transpose`, slicing, `astype` and
`clone` each return independently owned compact storage. So "view-based versus
materialized" is not a choice a caller can make, and the two cannot be
compared at the public layer. The non-compact layouts above are reached
through `Tensor._from_metadata`, and every such case says so.

This also bounds how much the non-compact gather cost matters today: it is a
**latent** cost, reachable only from inside the package, not one that ordinary
use pays.

**Truncated:** the non-compact size curve stops near 65,000 elements. Gathering
a non-compact layout costs roughly 2.7 µs per element, so a 1024×1024 sample
would take seconds and would dominate the run. The reason is recorded on every
skipped point.

## Operations

**Not present in the package, so not benchmarked:** sparse tensors, complex
dtypes, in-place fused operators, batched matrix multiplication with
broadcasting over leading dimensions beyond what `matmul` provides, pooling,
normalization layers with learned parameters, and any operation not exported
from `tensors/__init__.py`.

**Present but not swept as curves:**

- `slice_scatter` and `__setitem__` assignment. These mutate, so a calibrated
  loop measures a sequence of different states rather than one repeated
  operation. Storage invalidation — the part that has a performance
  consequence — is measured directly instead, as a `single_shot` case.
- `gradcheck`. It is a correctness tool built from repeated reverse passes,
  and its cost is the cost of those passes, which are measured directly.
- `Graph.release`, `uncompile`, `rebuild`. Lifecycle operations whose cost is
  dominated by the compilation they discard or redo, measured separately.

## Higher-order derivatives

First and second order are measured, along with full Jacobians and Hessians at
small widths (each runs one reverse pass per output element, so they are
quadratic in the wrong direction to sweep). Third order and beyond are not
measured: they depend on `backward_graph` being implemented for every
operation in the chain, which is not uniformly true, so a sweep would be
measuring which rules exist rather than how fast they are.

## Concurrency

Not benchmarked. The package keeps graph state per thread
(`tensors/graph/state.py`) and `Graph` keeps execution metadata per thread,
so multi-threaded throughput is a meaningful question — but answering it needs
a scaling study across thread counts with contention isolated from the GIL's
effects, which is a separate exercise from attributing single-threaded cost.
Stated here so its absence is deliberate.

## Memory measurement limits

- `tracemalloc` observes Python-level allocations. A NumPy array's data buffer
  is allocated by NumPy's own allocator and is **not** fully attributed, so
  `host_peak_bytes` understates large array allocations. It remains valid for
  comparing allocation *counts* and for host-side object churn, which is what
  it is used for here.
- Device memory is read from the CuPy pool, which is accurate for what the
  pool serves but does not see driver-level allocations made outside it.
- Per-call retention is measured after forcing collection. A negative figure
  means the collector reclaimed more than the batch allocated, and is reported
  rather than clamped.

## Single machine, single pass

Every figure comes from one machine and one pass, with five interleaved
samples per case. The design goal was breadth and attribution — the same
computation priced at every layer — not a distribution estimate per case.
High-variance cases are flagged and listed rather than averaged into
confidence they do not have.

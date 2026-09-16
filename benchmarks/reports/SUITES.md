# Benchmark suites and cases added

Twenty-one suites were added under `benchmarks/perfmap/suites/`. Case counts
come from the merged report; this file describes what each suite measures and
why it is shaped that way.

## Layer attribution

| Suite | What it isolates |
| --- | --- |
| `layers` | the full ladder — provider, kernel, dispatch, public, eager Variable, compiled replay — over binary arithmetic, unary transforms, activations, and comparisons, across the powers-of-ten size curve and four dtypes |
| `framework` | the machinery with the arithmetic removed: backend selection, kernel lookup (cold and cached), the workload policy, dtype resolution, shape and stride construction, broadcast agreement, result wrapping, operation/node/edge construction, Variable construction, operation recording, and fragment compilation |
| `reductions` | reductions at every depth, plus the stability guard's components measured on their own (`_summation_guard`, the `bool()` that consumes it, `_scaled_sum`), across full/one-axis/multi-axis reductions, `keepdims`, two dtypes, five sizes, and both same-sign and mixed-sign data |
| `linalg` | matrix products from 2×2 to 2048×2048, five rectangular shapes, and vector dot/norm/outer, with the matmul VJP beside the forward |
| `broadcasting` | nine broadcast patterns forwards, and the `sum_to_shape` / `sum_products_to_shape` reductions a reverse pass performs, against the provider reduction that does the same job |
| `shape` | transpose, reshape, concat, stack, cast, where, maximum, clip, slice, row indexing, and scalar indexing at provider, kernel, dispatch, and public depths |
| `creation` | zeros, ones, full, arange, linspace, eye across four dtypes and a size curve — operations with no input, so their whole cost is construction |
| `convolution` | 1-D/2-D/3-D spatial-extent curves plus a separate single-knob curve for batch, input channels, output channels, kernel size, stride, padding, dilation, and groups, both dtypes, forward and backward, and four realistic image-layer geometries |
| `losses` | cross-entropy and binary cross-entropy split into target preparation, the fused kernel, the public call, the VJP, and the differentiated public path |
| `random` | seeded and unseeded uniform, normal, and integer generation |
| `initializers` | seven sampling initializers plus orthogonal, over six parameter shapes and two dtypes |

## Layout, storage, and memory

| Suite | What it isolates |
| --- | --- |
| `layout` | the copy each public shape operation performs against the provider's view, and kernels reading contiguous / offset / transposed / strided / broadcast layouts built through the internal metadata constructor |
| `storage` | construction from list, `array.array`, provider array, provider Storage, owned Storage and Tensor; first versus cached representation lookup for all three kinds; direct conversion; copy; host materialization; dtype conversion; invalidation; scalar extraction three ways; and equality |
| `memory` | allocation and retention for ordinary operations, eager tracing, a persistent leaf's edge list, repeated replay, repeated reverse passes, the storage representation cache, and repeated training iterations |

## Graph, autograd, and execution modes

| Suite | What it isolates |
| --- | --- |
| `graph` | structural recording, then compilation split into dependency resolution, slot assignment, instruction emission and view resolution; plus plain-Tensor vs eager-Variable vs forward-replay vs reverse-replay over chain, wide, diamond and shared-subexpression topologies at 1–4,000 nodes and three widths; plus structural, nested, and functional `Graph` models |
| `autograd` | fourteen single-operation gradients across two dtypes and three widths; chain, branched and accumulation topologies at four depths; the components of a reverse pass (state validation, demand analysis, seed construction, accumulation at four fan-in counts, per-operation result validation); selective versus full backward; first and second order, Jacobian and Hessian; matmul, broadcast and convolution gradients |
| `fusion` | seven chain depths against four widths and two dtypes, comparing raw provider, plain Tensor, eager Variable, compiled replay with fusion declined, compiled replay with its fusion plan, the planning cost itself, and the fused reverse pass; plus an operand-count axis |
| `optimizer` | SGD, Adam and RMSprop over ten (parameter count × parameter size) configurations and two dtypes, split into first step, steady step, gradient preparation, gradient zeroing, and the single and batched update kernels |
| `training` | five model shapes × four batch sizes × two dtypes, each split into forward, loss, backward and optimizer phases plus the complete step and a ten-step run; and a retracing versus compiled-replay comparison |

## Cold start and CUDA primitives

| Suite | What it isolates |
| --- | --- |
| `startup` | thirteen fresh-interpreter measurements (interpreter baseline, NumPy/CuPy import, package import, backend probing, first Tensor, first operation per backend, CUDA context, first graph, first backward, first training step) plus in-process cold-versus-warm pairs for kernel lookup, provider-module resolution, and graph execution |
| `cuda` | the launch floor, idle stream and device barriers, launch-then-barrier, scalar reads, `bool()` of a reduction, the exact finite-result check the kernels perform, pooled and unpooled allocation, host↔device and device↔device transfers at four sizes, and the package's own transfer paths; plus the public operations whose kernels contain a host-visible check, at three sizes and both dtypes |

## Why the shapes are what they are

- **Size curves, not representative sizes.** Elementwise work sweeps
  1 → 10⁷ in powers of ten. The bottom of a curve is almost entirely fixed
  overhead; the top is where per-element cost dominates. A single size cannot
  separate them.
- **Both reduction data patterns.** The stability guard has a fast path for
  finite same-sign values, so a same-sign buffer and an alternating-sign
  buffer take different routes through the same kernel. Both are measured and
  tagged.
- **One knob at a time for convolution.** Batch and channels scale the matrix
  product; spatial extent and kernel size scale both the product and the
  im2col expansion; stride and padding change the output extent without
  changing the input. Varying them together would attribute cost to "a bigger
  shape" rather than to a knob.
- **Topology, not only depth, for graphs.** A chain, a wide fan-in, a diamond
  and a shared subexpression can compile to the same instruction count while
  exercising different traversal, reachability and accumulation work.
- **Parameter count separated from parameter size for optimizers.** The
  batched update kernels exist precisely so that many small parameters are not
  updated one at a time, so the two axes are swept independently.

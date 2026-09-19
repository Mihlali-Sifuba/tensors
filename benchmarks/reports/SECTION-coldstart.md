Cold-start costs are measured in **fresh interpreters** — one subprocess per
sample, timed end to end. That includes interpreter startup, which is why the
`interpreter_baseline` row exists: subtract it to get the cost of what the case
added. The subtraction is shown rather than baked in, so the raw figures stay
interpretable.

Each row is the median of five samples; because the same case is measured
under each backend context — the cost is a property of the environment, not
of the active backend — the last column names the contexts whose independent
readings agree.

### Cold: fresh interpreter per sample

| what a fresh interpreter did | wall clock | minus empty interpreter | worst MAD | readings agree across |
|---|---|---|---|---|
| `interpreter_baseline` | 67 ms | _baseline_ | 12.4% | cuda, numpy, python |
| `import_numpy` | 230 ms | **162 ms** | 10.7% | cuda, numpy, python |
| `import_cupy` | 708 ms | **641 ms** | 3.1% | cuda |
| `import_tensors` | 243 ms | **176 ms** | 9.6% | cuda, numpy, python |
| `available_backends` | 882 ms | **815 ms** | 5.9% | cuda, numpy, python |
| `first_tensor_construction` | 253 ms | **186 ms** | 7.6% | cuda, numpy, python |
| `first_python_operation` | 261 ms | **194 ms** | 8.5% | cuda, numpy, python |
| `first_numpy_operation` | 404 ms | **337 ms** | 9.1% | cuda, numpy, python |
| `cuda_context_only` | 793 ms | **726 ms** | 0.3% | cuda |
| `first_cuda_operation` | 958 ms | **891 ms** | 3.8% | cuda |
| `first_graph_execution` | 382 ms | **314 ms** | 4.6% | cuda, numpy, python |
| `first_backward` | 409 ms | **342 ms** | 2.0% | cuda, numpy, python |
| `first_training_step` | 409 ms | **342 ms** | 7.5% | cuda, numpy, python |

### Warm: the same interpreter, caches cleared

| pair | backend | cold | warm | cold / warm |
|---|---|---|---|---|
| trace, compile and run a graph vs replay it | python | 563.22 us | 214.15 us | 2.6x |
| trace, compile and run a graph vs replay it | numpy | 204.76 us | 33.33 us | 6.1x |
| trace, compile and run a graph vs replay it | cuda | 370.58 us | 103.58 us | 3.6x |
| kernel lookup after a cache clear vs cached | numpy | 1.28 us | 216 ns | 5.9x |
| kernel lookup after a cache clear vs cached | cuda | 1.18 us | 208 ns | 5.6x |
| provider-module resolution after a cache clear vs cached | numpy | 1.81 us | 936 ns | 1.9x |
| provider-module resolution after a cache clear vs cached | cuda | 1.89 us | 929 ns | 2.0x |

### What these say

- **Importing the package costs about as much as importing NumPy** (176 ms
  against 162 ms), despite `tensors` having no runtime dependencies.
  `tensors/__init__.py` imports every subpackage eagerly, so a program that
  only wants a `Tensor` pays for `graph`, `optim`, `init`, `random` and the
  whole `math` surface.
- **`available_backends()` costs 815 ms on a cold interpreter**, because
  `_cuda_available()` imports CuPy and asks the driver for a device count.
  Anything that probes backends at import time — including
  `use_backend("cuda")`, which resolves through the same predicate — pays
  this once.
- **CUDA's first operation is dominated by context creation**, not by the
  package: 726 ms of the 891 ms is `cupy.cuda.Device().synchronize()` on its
  own.
- **The first differentiated step is not much more expensive than the first
  forward** (342 ms against 314 ms): cold start is dominated by imports, not
  by autograd.
- **Warm caches matter far more than cold ones.** Clearing the kernel-lookup
  cache costs about 1 µs on the next lookup, which is small — but note that
  entering *any* `use_backend` scope clears it, so a program that switches
  backend contexts in a loop pays it repeatedly.
- **Replay is 2.6× to 6.1× cheaper than tracing the same graph**, in a warm
  interpreter. That is the clearest single argument for the compiled path.

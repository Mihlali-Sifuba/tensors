# Benchmarks

These benchmarks establish repeatable local performance baselines for the public
`tensors` API. They are intended to reveal regressions and guide optimization;
they are not correctness tests or claims about performance on other machines.

## Run the suite

From the repository root:

```powershell
python -m benchmarks --quick
```

The quick run is useful during development. The default run spends longer
calibrating and sampling each case:

```powershell
python -m benchmarks
```

Select a group or filter case names when investigating a particular subsystem:

```powershell
python -m benchmarks --suite graph
python -m benchmarks --match matmul
python -m benchmarks --list
```

Save the complete measurements and environment metadata for comparison:

```powershell
python -m benchmarks --output benchmark-results.json
```

## What is measured

| Suite | Cases |
| --- | --- |
| `tensor` | elementwise arithmetic, broadcasting, reductions, matrix multiplication, and norms |
| `graph` | constructing a fresh `Graph` and replaying an already traced `Computation` |
| `autograd` | `grad`, `backward`, derivative-graph construction, second derivatives, and Hessians |
| `training` | a complete MLP step: forward pass, loss, gradient reset, backward pass, and SGD update |

Graph construction and replay are deliberately separate. Construction measures
the cost of expressing and tracing a function; replay measures repeated evaluation
of the computation that representation produced. Likewise, ordinary gradients are
kept separate from `create_graph=True` and Hessian cases because higher-order
differentiation builds additional graph structure.

## Methodology

Each case validates its result before timing. Stable inputs and models are created
outside the timed callable unless their construction is the operation being
measured. The runner calibrates an iteration count, takes repeated samples, and
reports the median, minimum, and median absolute deviation (MAD).

Python's cyclic garbage collector stays disabled for ordinary tensor kernels, as
it is under `timeit`. It is enabled for cases that intentionally construct graph
objects, because those objects can contain cycles and collection is part of their
sustained cost.

Compare results only under similar conditions. Use the JSON report's Python,
platform, processor, Git commit, and dirty-worktree metadata, and avoid running
unrelated heavy workloads at the same time. A useful regression check compares
several runs from the same machine rather than treating a single sample as
definitive.

## Adding a case

Add a `BenchmarkCase` to the relevant `*_cases.py` module. Keep setup outside the
timed function, provide a validation callback, and set `work_items` only when the
unit of work is meaningful enough to report as throughput. Enable garbage
collection when a case creates cyclic graph structures on every iteration.

The runner intentionally uses only the Python standard library so benchmarks work
in a normal development checkout without an additional benchmarking dependency.

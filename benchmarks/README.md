# Benchmarks

These benchmarks establish repeatable local performance baselines for `tensors`.
They are designed to locate overhead, identify backend crossover points, and show
whether an optimization improved the intended layer. They are not correctness
tests or performance claims for other machines.

## Run the benchmarks

From the repository root, run the compact development baseline with:

```powershell
python -m benchmarks --quick
```

The default `core` suite retains the original tensor, graph, autograd, and small
training baselines. `--quick` takes three short samples; omitting it takes seven
longer samples:

```powershell
python -m benchmarks
```

The core suite is a latency and regression baseline, not a complete backend
ranking. Most of its tensors and matrices are deliberately small, so NumPy will
often beat CUDA by avoiding kernel-launch and synchronization costs. CUDA should
be judged with size curves, wide graph replay, and sufficiently large or batched
optimizer work while values remain device-resident.

The extended suites are deliberately opt-in because some cases allocate millions
of values or exercise currently expensive backward paths:

```powershell
python -m benchmarks --suite scaling
python -m benchmarks --suite storage
python -m benchmarks --suite autograd --match matrix_backward
python -m benchmarks --suite autograd --match planned_chain
python -m benchmarks --suite optimizer --match _many
python -m benchmarks --suite all
```

For a focused NumPy/CUDA crossover investigation, run:

```powershell
python -m benchmarks --backend accelerated --suite scaling
python -m benchmarks --backend accelerated --suite graph --match width-100000
python -m benchmarks --backend accelerated --suite optimizer
```

By default, each eligible case runs on Python and every installed optional
backend. Select one backend or all installed accelerators when narrowing an
investigation:

```powershell
python -m benchmarks --backend python --suite scaling
python -m benchmarks --backend numpy --suite provider
python -m benchmarks --backend cuda --suite storage
python -m benchmarks --backend accelerated --suite chain
```

NumPy and CUDA require their respective optional dependencies. `--backend auto`
selects NumPy when available and Python otherwise. Use `--list` to see both case
names and their eligible backends without timing them:

```powershell
python -m benchmarks --suite all --list
```

Save measurements and environment metadata for a before-and-after comparison:

```powershell
python -m benchmarks --backend accelerated --suite scaling --output before.json
python -m benchmarks --backend accelerated --suite scaling --output after.json
```

## Suites

| Suite | What it isolates |
| --- | --- |
| `core` | compact historical regression baseline used by the default command |
| `tensor` | primitive public tensor operations at representative sizes |
| `backend` | broader public unary, normalization, loss, layout, creation, and optimizer coverage |
| `provider` | raw NumPy or CuPy primitives versus guarded internal backend kernels |
| `scaling` | public elementwise, reduction, broadcast, and matrix-multiplication size curves |
| `storage` | host/device conversion, cached lookup, materialization, and mutation invalidation |
| `chain` | unfused expression chains across tensor width and expression depth |
| `graph` | trace versus compiled replay for scalar-chain, branch, and matrix-heavy graph topologies |
| `autograd` | forward, backward, compiled deep-chain VJPs, accumulation, matrix gradients, graph creation, and higher derivatives |
| `loss` | target preparation, dense backend kernel, public cross-entropy, and backward pass |
| `optimizer` | first-step state creation, steady updates, and many-small-parameter batching for SGD, Adam, and RMSprop |
| `training` | complete MLP steps and separate forward, replay, loss, backward, and optimizer phases |
| `startup` | optional-provider import and first tensor operation in fresh interpreters |
| `all` | every suite above; intended for deliberate, long-running investigations |

## Reading the layers

Every JSON result records a `layer`. Comparing adjacent layers explains where
time is spent:

1. `provider` measures NumPy or CuPy directly.
2. `kernel` adds the internal backend guard and storage contract.
3. `public` adds tensor dispatch and result construction.
4. `graph` and `autograd` add tracing and differentiation.
5. `optimizer` and `training` measure complete algorithmic phases.

`storage` and `startup` isolate conversion and initialization costs separately.
For example, if raw provider and kernel matrix multiplication are close but the
public operation is slower, the likely target is public dispatch or storage—not
the numerical kernel.

Not every case is meaningful on every backend. Accelerator-only rows display `-`
for Python in comparison tables, and no speedup is calculated without a matching
Python measurement.

## Methodology

Each case validates its result before timing. Stable inputs and models are created
outside the timed callable unless their creation is explicitly part of the case.
The runner calibrates an iteration count, takes repeated samples, and reports the
median, minimum, and median absolute deviation (MAD).

Validation and calibration warm ordinary cases. A name containing `first` means
that fresh storage, model, or optimizer state is created inside each timed call;
only the `startup` suite measures a genuinely fresh interpreter. Optimizer
`steady` cases reuse initialized state.

CUDA operations are asynchronous, so the active CuPy stream is synchronized after
every timed iteration. CUDA durations therefore include completed device work,
not just Python-side kernel launch time.

Python's cyclic garbage collector stays disabled for ordinary kernels, as it is
under `timeit`. It is enabled for cases that intentionally construct cyclic graph
objects so collection remains part of their sustained cost.

Compare results only under similar conditions. JSON reports include backend,
Python, platform, processor, Git commit, dirty-worktree state, and CUDA device
metadata. Prefer several runs on the same machine over treating one sample as
definitive, and investigate a high MAD before drawing conclusions.

## Adding a case

Add a `BenchmarkCase` to the relevant `*_cases.py` module. Keep unrelated setup
outside the timed callable, provide a validation callback, assign the closest
`layer`, and restrict `backends` when a case is accelerator-specific. Use a size
curve rather than one arbitrary size when looking for a crossover point, and split
end-to-end work into phases when attribution matters.

Set `work_items` only when its unit is meaningful as throughput. Enable garbage
collection when each invocation creates cyclic graph structures. The runner uses
only the Python standard library, so no separate benchmark dependency is needed.

# perfmap — a layered performance map of `tensors`

`perfmap` measures the same computations at several depths of the execution
stack so that overhead can be **attributed to a layer** rather than merely
observed. It is measurement only: it imports the package under test and never
modifies it.

The original `benchmarks` suite remains available and unchanged
(`python -m benchmarks`). `perfmap` is a separate, broader harness.

## Run it

```powershell
# everything (long; prefer the staged form below)
python -m benchmarks.perfmap --output benchmarks/results/perfmap.json --memory

# one suite
python -m benchmarks.perfmap --suite reductions --output out/reductions.json

# one narrow question
python -m benchmarks.perfmap --suite layers --match "binary/add/float64" --backend cuda

# what exists
python -m benchmarks.perfmap --list-suites
python -m benchmarks.perfmap --suite fusion --list-groups
```

Staged, which is how the baseline was produced:

```bash
bash benchmarks/results/run_stages.sh framework cuda layout shape storage ...
python -m benchmarks.perfmap.merge "benchmarks/results/baseline/*.json" \
    --output benchmarks/results/perfmap.json
```

Merging re-derives every analysis table from the combined records, so a ladder
whose rungs come from different suites still resolves.

## Layers

A case declares the layer it measures. Comparing adjacent layers over the same
computation is the whole point.

| Layer | What it measures |
| --- | --- |
| `provider` | `numpy.*` / `cupy.*` directly, on native arrays |
| `kernel` | the internal guarded kernel, called with Tensors |
| `dispatch` | `execute_*`: workload policy plus kernel lookup |
| `public` | the public `Tensor` operation |
| `variable` | eager `Variable`: records structure, compiles a fragment, runs it |
| `graph-trace` | recording a graph |
| `graph-compile` | turning a graph into an instruction program |
| `graph-replay` | running an already-compiled program |
| `autograd` | reverse passes and their components |
| `optimizer` | optimizer steps and phases |
| `training` | complete steps and their phases |
| `storage` | construction, conversion, caching, materialization |
| `fusion` | fused execution and planning |
| `sync` | CUDA launch, barrier, transfer, and allocation primitives |
| `startup` | cold start, and cold-versus-warm caches |
| `memory` | cases that exist for the allocation pass |

## How comparisons are wired

Cases opt into a derived table by carrying a **tag**:

- `ladder` — every layer measuring the same computation, dtype and size shares
  one ladder key. `analysis.ladders` resolves the ratio and added time between
  each neighbouring pair, and names the dominant step.
- `curve` — points on one size curve, used for scaling and crossover analysis.
- `dtype_pair` — matches a float32 case to its float64 twin.
- `pair` + `phase` — matches forward/backward, trace/replay, and
  cold/warm pairs.
- `layout` — matches a strided layout to its contiguous baseline.
- `data` — `same-sign` versus `mixed-sign`, which take different routes
  through the reduction stability guard.

## Methodology

**Interleaving.** A measurement is a `(case, backend)` job. Jobs are grouped so
that only comparable work is alive at once, and within a group every backend's
jobs are measured in **rotated, seeded-random order across several rounds**, one
sample per round. A thermal or allocator drift therefore spreads across all
jobs instead of landing on whichever backend ran last. `--seed` fixes the order.

**Grouping** also bounds live memory: one group's inputs exist at a time, and
the CuPy pool is released between groups.

**Warm-up.** Every job is validated and calibrated, and each sample takes one
untimed call after entering the backend context — necessary because entering a
scoped backend context clears the kernel-lookup cache.

**Calibration.** An invocation count is chosen so a batch approaches
`--target-time`. Cases that perform one state transition per sample declare
`single_shot`, so calibration cannot make their state depend on backend speed.

**Synchronization (CUDA).** Sync is inserted only at batch boundaries, never
between measured calls, so an implementation that does not synchronize is never
charged for one. Three numbers are recorded:

- `host_submit` — host time to return from the calls;
- `host_total` — host time until the device finished (the comparison statistic);
- `device` — GPU execution time from CUDA events.

**Detecting hidden synchronization.** Comparing submit against total cannot do
it: a call that merely launches many kernels is host-bound for the same reason
a blocking one is. Instead, `probe_synchronization` occupies the device with a
measured amount of queued work and then issues one call. Queued work is
asynchronous, so only a call that synchronizes has to absorb it.
`absorbed_fraction_of_barrier` near 1 means the call blocks. The probe is
validated against known-async (`cupy.add`) and known-blocking
(`bool(cupy.any(...))`) references.

**Statistics.** Median (the comparison statistic), MAD, min, max, mean, stdev,
p95, and every raw sample are retained. A case whose MAD exceeds 15% of its
median is flagged `noisy` and listed rather than quietly averaged.

**Garbage collection** is disabled during timing, except for cases that build
cyclic graph objects each call — there collection is part of the path's
sustained cost, so it stays in scope.

**Memory** is measured in a **separate untimed pass** (`--memory`), because
tracing allocations perturbs timing badly enough to corrupt both. Cases marked
`memory` report what one call allocates, how many allocations it makes, and what
a batch of calls fails to release.

**Unsupported combinations are recorded, never omitted.** A factory raises
`Unsupported` with a stated reason, and the record carries
`classification="unsupported"` plus that reason. Size ceilings are declared per
backend in `workloads.py` for the same reason.

## Outputs

`--output x.json` writes three files:

| File | Contents |
| --- | --- |
| `x.json` | metadata, settings, every record, and all derived analysis tables |
| `x.csv` | one flat row per record — the schema future branches are joined on |
| `x-samples.csv` | every retained raw sample, one row each |

Metadata records commit, dirty state and dirty paths, branch, upstream commit,
OS, Python, NumPy (and its BLAS), CuPy, CUDA runtime and driver, GPU name,
compute capability, SM count and memory, CPU, and the synchronization policy.

## Adding a case

Add it to the relevant `suites/*.py`, from a factory that receives the active
backend. Build shared inputs in the factory (not in `run`), give it a
`validate`, assign the closest `layer`, and tag it into whichever comparison it
belongs to. Raise `Unsupported` with a real reason rather than returning
nothing. Use `reset` when the timed callable mutates state, `single_shot` when a
sample is one transition, `gc_enabled` when each call builds cyclic objects, and
`memory=True` when allocation behavior is part of the question.

Draw sizes and dtypes from `workloads.py` so a curve means the same thing
everywhere, and prefer a curve to a single representative size.

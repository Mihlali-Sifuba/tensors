# Benchmarks — a layered performance map of `tensors`

These benchmarks measure the same computations at several depths of the
execution stack, so that overhead can be **attributed to a layer** rather than
merely observed. They are measurement only: the package under test is
imported, never modified.

There is one benchmark system. One case model, one runner, one timing and
synchronization policy, one statistics implementation, one registry, one
command, and one result schema.

## How it is organized

A suite answers one of three questions, and its directory says which.

| Directory | Question | Examples |
| --- | --- | --- |
| `workloads/` | What does this operation cost, and where does that cost enter? | `arithmetic`, `reductions`, `linalg`, `convolution` |
| `execution/` | What does the machinery around an operation cost on its own? | `graph`, `backward`, `fusion`, `storage`, `startup` |
| `scenarios/` | What does a whole workflow cost, every layer included? | `training`, `optimizer` |

Workloads are organized by what an operation **means**, never by backend: one
workload definition runs on Python, NumPy, and CUDA. A workload also owns its
**complete ladder** — the same computation measured at the provider, the
guarded kernel, dispatch, the public operation, an eager `Variable`, and a
replayed graph — and those rungs stay in one group, so the difference between
two of them is overhead rather than a difference of operand or sampling round.

Two more directories hold things a suite uses rather than measures:

- `baselines/` — direct access to NumPy and CuPy, called on their own terms.
  This is deliberately separate from selecting a `tensors` backend: the point
  of a baseline is to show what the Tensor, the dtype decision, the dispatch,
  and the storage wrapper cost by leaving them out.
- `scenarios/models/` — the models a scenario runs, so a suite measures a
  model rather than defining one.

## Run it

```powershell
# a fast check that the harness and the suites still work
python -m benchmarks --profile quick

# the full matrix
python -m benchmarks --profile standard --output out/map.json

# one suite, one question
python -m benchmarks --suite arithmetic --match "add/float64" --backend cuda

# what exists
python -m benchmarks --list-suites
python -m benchmarks --suite fusion --list-groups

# what would be measured, without measuring it
python -m benchmarks.inventory --summary
```

Staged, which is how the baseline was produced:

```bash
bash benchmarks/reports/run_stages.sh dispatch synchronization layout ...
python -m benchmarks.reporting.merge "benchmarks/reports/baseline/*.json" \
    --output benchmarks/reports/map.json
```

Merging re-derives every analysis table from the combined records, so a ladder
whose rungs come from different suites still resolves.

## Profiles

A workload says *what* to measure and at which sizes the question is
interesting. A profile says how much of that to run today. The two are kept
apart because they change for different reasons: a size belongs in a curve
because the cost changes there, and it is skipped because there is no time.

| Profile | Rounds | Sample | Selection |
| --- | --- | --- | --- |
| `quick` | 3 | 10 ms | both ends of each curve, float64 only, tight ceilings |
| `standard` | 5 | 50 ms | everything the suites declare |
| `comprehensive` | 9 | 100 ms | everything, longer, plus the allocation pass |

A profile **filters** a declared curve rather than supplying one, so no
profile can introduce a case a suite did not declare, and `standard` runs
exactly the full matrix. `--rounds` and `--target-time` override the chosen
profile explicitly.

Memory is a dimension of a run, not a suite: a case declares `memory=True`,
and `--memory` or the `comprehensive` profile turns the separate allocation
pass on.

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
| `startup` | cold start, and cold-versus-warm caches |
| `memory` | cases that exist for the allocation pass |
| `sync` | CUDA launch, barrier, transfer, and allocation primitives |

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
backend in `inputs.py` for the same reason, and a profile may tighten
them further for a particular run.

## Outputs

A report carries `"schema": "tensors-perfmap/2"`. Version `/1` labelled
a record with the suite that produced it under the earlier flat layout,
where one `layers` suite covered arithmetic, the elementary functions,
the activations, and the comparisons together. The timings, samples,
layers, and families did not change, so a `/1` report is still readable
and the merge path accepts both — but its `suite` and `group` strings
name a different vocabulary, and a merge across versions records that
rather than smoothing it over.

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

Add it to the module that owns the question: a semantic domain under
`workloads/`, framework behaviour under `execution/`, a whole workflow
under `scenarios/`. Write it as a factory that receives the active
backend. Build shared inputs in the factory (not in `run`), give it a
`validate`, assign the closest `layer`, and tag it into whichever comparison it
belongs to. Raise `Unsupported` with a real reason rather than returning
nothing. Use `reset` when the timed callable mutates state, `single_shot` when a
sample is one transition, `gc_enabled` when each call builds cyclic objects, and
`memory=True` when allocation behavior is part of the question.

Draw shared curves, dtypes, ceilings, and input builders from
`inputs.py`, and wrap a curve in `profiles.selected_sizes` so a profile
can narrow it. Prefer a curve to a single representative size.

Geometry that belongs to one workload stays with it: convolution
shapes, matrix aspect ratios, and model dimensions are part of the
question that workload asks, not shared configuration.

Before and after any structural change, compare
`python -m benchmarks.inventory` against itself. It lists every case
the suites would build, with its backend, layer, family, dtype, shape,
element count, tags, and supported status, so a move that was meant to
be structural can be shown to have been.

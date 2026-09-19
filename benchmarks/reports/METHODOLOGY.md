# Baseline methodology

This records exactly how the `perfmap` baseline was produced, so a later
optimization branch can be compared against it on equal terms.

## Subject

| Item | Value |
| --- | --- |
| Repository | `Mihlali-Sifuba/tensors` |
| Baseline commit | `04e8dd252ed9aadd2737cbf59ed3c7a80d9ab9ed` (`main`) |
| Working tree at baseline | clean; `main == origin/main` |
| Measurement branch | `perf/comprehensive-benchmarking` |
| Production code changed | none — nothing under `tensors/` was modified |

The measurement branch adds `benchmarks/perfmap/` and `benchmarks/reports/`
only. The pre-existing `benchmarks/` suite is untouched and still runs as
`python -m benchmarks`.

## Command

Suites were run one at a time so that a long run could be monitored and a
single failure would not discard the rest:

```bash
ROUNDS=5 TARGET=0.02 bash benchmarks/reports/run_stages.sh <suite> ...
python -m benchmarks.reporting.merge "benchmarks/reports/baseline/*.json" \
    --output benchmarks/reports/perfmap.json
python -m benchmarks.reporting.profile_report \
    --output benchmarks/reports/profiling.json
```

Per-suite settings: 5 measured samples per case (one per interleaved round),
a 20 ms calibration target per sample, the seeded default execution order,
and the separate memory pass enabled (`--memory`).

## What one measurement is

A measurement is a `(case, backend)` job. A case declares the layer it
measures, and the same computation is measured at every layer it can be
isolated at, so the difference between two adjacent layers is the cost of the
layer between them.

## Ordering, and why

Jobs are collected into **groups** of comparable work. Within a group, every
backend's jobs are measured in **rotated, seeded-random order across five
rounds**, one sample per round. Three properties follow:

- comparable cases across backends are measured within seconds of each other,
  not in three separate phases;
- a thermal, allocator, or clock drift during the run spreads across all jobs
  instead of landing on whichever backend was measured last;
- the order is reproducible from `--seed`.

Grouping also bounds live memory: one group's inputs exist at a time, and the
CuPy pool is released between groups.

## Warm-up and calibration

Each job is validated, then calibrated to choose an invocation count whose
batch approaches the 20 ms target. Each sample additionally performs one
untimed call after entering the backend context — necessary because entering a
scoped backend context clears the kernel-lookup cache.

Cases that perform a single state transition per sample (a first optimizer
step, a storage invalidation, a fresh-interpreter start) declare `single_shot`,
so calibration cannot batch several transitions into one sample and make the
input state depend on backend speed.

## Synchronization policy (CUDA)

Synchronization is inserted **only at batch boundaries, never between the
measured calls**, so an implementation that does not synchronize is never
charged for one. Three numbers are recorded per case:

| Metric | Question it answers |
| --- | --- |
| `host_submit` | how long the host spent issuing the work |
| `host_total` | how long until the device finished it (**the comparison statistic**) |
| `device` | how long the GPU executed, from CUDA events |

### Detecting hidden synchronization

Comparing submission against completion cannot detect a barrier: a call that
merely launches many kernels is host-bound for the same reason a blocking one
is, and once enough launches are queued the device throttles submission
regardless.

Instead, each CUDA job is probed once: the device is occupied with a measured
amount of queued work, then one call is issued. Queued work is asynchronous, so
a non-blocking call returns while it is still running and only a call that
synchronizes has to absorb it. `absorbed_fraction_of_barrier` near 1 means the
call blocks.

The probe was validated against references with known behavior:

| Reference | Absorbed | Verdict |
| --- | --- | --- |
| `cupy.add` (asynchronous) | ~0% | does not block ✔ |
| `bool(cupy.any(...))` (blocking) | 95% | blocks ✔ |
| `float(array[0])` (blocking) | 96% | blocks ✔ |

## Statistics

Median (the comparison statistic), MAD, min, max, mean, standard deviation,
p95, and **every raw sample** are retained. A case whose MAD exceeds 15% of its
median is flagged `noisy` and listed in its own table rather than quietly
averaged.

## Garbage collection

Disabled during timing, except for cases that construct cyclic graph objects on
every call. There, collection is part of the path's sustained cost, so it stays
in scope; those cases are marked `gc_enabled`.

## Memory

Measured in a **separate, untimed pass**, because tracing allocations perturbs
timing badly enough to corrupt both. For each case marked `memory`:

- one traced call reports what the operation allocates and how many
  allocations it makes (`tracemalloc`, plus CuPy pool deltas on CUDA);
- a batch of calls, compared before and after with collection forced, reports
  what the operation fails to release — its per-call retention.

## Unsupported combinations

Nothing is silently omitted. A case factory raises `Unsupported` with a stated
reason, and the record carries `classification="unsupported"` plus that reason.
Per-backend size ceilings are declared in `workloads.py` and reported the same
way, so a missing point always has an explanation attached.

## Profiling

Profiling is **diagnostic only** and reported separately from measurements. It
uses deliberately intrusive instrumentation:

- the provider module the kernels call is replaced by a counting proxy, giving
  an exact count of which provider primitives one public operation invokes;
- CuPy's host-conversion methods are wrapped, giving an exact count of
  device-to-host conversions per call;
- `cProfile` attributes fixed overhead to Python functions.

No timing from a profiled run is comparable to a benchmark result.

## Known methodological caveats

1. **Laptop GPU under WDDM.** Kernel launch costs about 20 µs and an idle
   stream synchronize about 0.4 µs, but a barrier after real work costs
   roughly 100–135 µs. Absolute CUDA figures are specific to this machine;
   the ratios and the blocking verdicts are not.
2. **CUDA variance.** Several CUDA cases show 15–25% MAD even across
   interleaved rounds. They are flagged and listed; conclusions resting on a
   single flagged row are provisional.
3. **One machine, one run.** Every figure comes from a single machine and a
   single pass. The design goal was breadth and attribution, not
   distribution estimates.
4. **Concurrency incident, and what was done about it.** During an early
   attempt, two staged pipelines were left running concurrently: stopping a
   background pipeline killed its shell but did not reap the measuring
   process beneath it, so an orphaned suite kept a core busy — and its
   parent loop kept advancing through further suites, writing the same output
   files. Any measurement taken while that was true is contended and
   therefore invalid.

   Every result from those attempts was **discarded**, not corrected. The
   baseline in this directory comes from a single pipeline started after
   verifying that no other Python or measuring process existed. Two
   safeguards were added:

   - `run_stages.sh` now refuses to start if any Python process is already
     running, and writes to a log file instead of through a pipeline, so no
     reader can exit and leave a writer orphaned;
   - the run is detached with `nohup`, so monitoring it cannot kill it and
     stopping a monitor cannot orphan it.

   The lesson is recorded here rather than smoothed over: a benchmark harness
   that can be silently contended produces numbers that look fine.
5. **Non-contiguous curve truncated.** Gathering a non-compact layout costs
   about 2.7 µs per element (a Python-level walk over logical coordinates), so
   that curve stops at roughly 65,000 elements. The reason is recorded on
   every skipped point.

6. **First-suite settling.** Interleaving equalizes *backends within a group*,
   but it cannot equalize *position within the run*. The first suite measured
   showed roughly double the medians and 30% MAD on its smallest cases
   compared with a later reading of the same cases — the machine was still
   settling. The two cheapest suites (`framework`, `cuda`) were therefore
   **re-measured at the end of the run**, in steady state, and both readings
   are reported: the steady-state one is used for analysis and the early one
   is kept as evidence of the drift. Any case whose two readings disagree
   materially is treated as low confidence.

7. **Two size ceilings found during the run, and why they exist.** Two paths
   turned out to be per-element Python work at every size, which made their
   largest points cost minutes per sample without adding a finding:

   - **integer creation** — the NumPy kernels build integer values in
     `object` dtype (`numpy.full(..., dtype=object)`, `numpy.arange(...,
     dtype=object)`), so a ten-million-element integer array is ten million
     Python ints; the CUDA kernels decline integers outright and fall back to
     the host reference implementation;
   - **integer elementwise arithmetic** — the same `object`-dtype widening in
     `_operand`, and the same CUDA refusal.

   Both are capped at one million values, and every skipped point carries the
   reason. The `creation` suite was stopped mid-run when this was found and
   re-measured afterwards with the ceiling in place, so no result mixes the
   two configurations.

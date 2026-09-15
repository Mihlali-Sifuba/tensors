# `tensors` performance map — baseline at `04e8dd25`

A layered performance map of the whole package: the same computations measured
at every depth of the execution stack, so that cost can be **attributed to a
layer** rather than merely observed.

Nothing under `tensors/` was modified. This is measurement, and the ranked
findings at the end are candidates for investigation, not changes made.

---

## 1. Exact commit and environment

### Subject

| Item | Value |
| --- | --- |
| Baseline commit | `04e8dd252ed9aadd2737cbf59ed3c7a80d9ab9ed` |
| `git describe` | `v0.6.0-67-g04e8dd2` |
| Branch measured from | `perf/comprehensive-benchmarking` |
| `main == origin/main` | yes, verified before starting |
| Working tree | production tree **clean**; only `benchmarks/perfmap/` and `benchmarks/results/` added |
| Production code changed | **none** — `git status --porcelain -- tensors/` is empty |

The recorded `git.dirty_paths` in every result file is exactly
`["benchmarks/perfmap/", "benchmarks/results/"]`, which is the machine-readable
form of that guarantee.

### Machine

| Item | Value |
| --- | --- |
| OS | Windows 11 Pro, `Windows-11-10.0.26200-SP0` |
| CPU | Intel Core i7-12700H — 14 cores / 20 threads |
| Host RAM | 31.75 GB total, ~6 GB free during the run |
| GPU | NVIDIA GeForce RTX 3050 Ti Laptop GPU |
| GPU compute capability | 8.6, 20 SMs, 1035 MHz |
| GPU memory | 4.29 GB total, 3.45 GB free; 128-bit bus @ 5501 MHz |
| Python | CPython 3.14.6 |
| NumPy | 2.5.2, built against `scipy-openblas` |
| CuPy | 14.2.0 (`cupy-cuda13x`) |
| CUDA runtime / driver | 13.2 / 13.0 |
| Backends available | `python`, `numpy`, `cuda` |

Two hardware facts matter for reading the CUDA numbers:

- **This GPU's FP64 throughput is a small fraction of its FP32 throughput**
  (consumer Ampere is 1/32). Any result where float32 is *slower* than float64
  is therefore a library effect, not a hardware one.
- **It is a laptop GPU under the Windows WDDM driver model.** Kernel launch
  costs about 20 µs and an idle stream synchronize about 0.4 µs, but a barrier
  after real work costs roughly 100–135 µs. Absolute CUDA latencies are
  specific to this machine; the ratios, the barrier counts, and the blocking
  verdicts are not.

---

## 2. Every benchmark suite and every case added

See `SUITES.md` for the full description of what each suite measures and why it
is shaped that way, and the *Inventory* tables below for the counts actually
recorded.

Twenty-one suites were added under `benchmarks/perfmap/suites/`. The harness
itself is `benchmarks/perfmap/` and is documented in its own `README.md`. The
pre-existing `benchmarks/` suite is untouched and still runs as
`python -m benchmarks`.

---

## 3. Exact benchmark methodology

The full statement is in `METHODOLOGY.md`. The parts that change how results
should be read:

**A measurement is a `(case, backend)` job**, and a case declares the layer it
measures. The same computation is measured at every layer it can be isolated
at, so the difference between adjacent layers is the cost of the layer between
them.

**Ordering.** Jobs are grouped by comparable work; within a group every
backend's jobs are measured in rotated, seeded-random order across five
rounds, one sample per round. Comparable cases across backends are therefore
measured within seconds of each other, and any drift during the run spreads
across all of them instead of landing on whichever backend ran last.

**CUDA synchronization.** Sync is inserted only at batch boundaries, never
between measured calls. Three numbers are recorded: host submission time, host
time to completion (the comparison statistic), and device execution time from
CUDA events.

**Detecting hidden synchronization.** Comparing submission against completion
cannot do it — a call that merely launches many kernels is host-bound for the
same reason a blocking one is. Instead the device is occupied with a measured
amount of queued work and one call is issued; queued work is asynchronous, so
only a call that synchronizes has to absorb it. The probe was validated
against `cupy.add` (asynchronous, ~0% absorbed), `bool(cupy.any(...))`
(blocking, 95%) and `float(array[0])` (blocking, 96%).

**Statistics.** Median, MAD, min, max, mean, stdev, p95 and every raw sample
are retained. A case whose MAD exceeds 15% of its median is flagged and listed
in its own table rather than quietly averaged.

**Memory** is measured in a separate untimed pass, because tracing allocations
perturbs timing badly enough to corrupt both.

**Unsupported combinations are recorded with reasons, never omitted.**

**One methodological failure is recorded rather than smoothed over.** During an
early attempt two staged pipelines were left running concurrently — stopping a
background pipeline killed its shell but did not reap the measuring process
beneath it, and its parent loop kept advancing through further suites. Every
result from those attempts was **discarded**; the baseline here comes from a
single pipeline started after verifying no other Python process existed, with
two safeguards added to the runner. See `METHODOLOGY.md` § Known
methodological caveats.

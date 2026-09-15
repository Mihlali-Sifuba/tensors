Everything measured is stored in machine-readable form so a future
optimization branch can be compared against this baseline by joining on the
same keys.

| File | Contents |
| --- | --- |
| `perfmap.json` | the merged baseline: metadata, settings, **every record**, and every derived analysis table |
| `perfmap.csv` | one flat row per record — the schema to join future runs on |
| `perfmap-samples.csv` | **every retained raw sample**, one row per sample per metric |
| `profiling.json` | diagnostic profiling: provider-call counts, device-to-host transfer counts, and cProfile attribution |
| `baseline/<suite>.json` / `.csv` / `-samples.csv` | the per-suite reports the merge was built from |
| `baseline/early/` | the first readings of the suites that were re-measured, kept as evidence of run-position drift |
| `baseline/run.log` | the full run log, including per-suite wall clock and exit status |

### The record schema

Each record carries `name`, `backend`, `suite`, `group`, `layer`, `family`,
`dtype`, `shape`, `elements`, `classification`, `loops_per_sample`, the
`tags` that place it in each comparison, and:

- `host_total` — median, mean, stdev, min, max, p95, MAD, variability percent,
  a `noisy` flag, sample count, and **the raw samples**;
- `host_submit` — the same statistics for host submission time;
- `device` — the same statistics for CUDA event time, where applicable;
- `cuda` — device share of latency, host overhead, the single-call probe, the
  barrier-absorption probe, and the `hidden_synchronization` verdict;
- `memory` — host peak bytes, host allocation count, per-call host retention,
  device pool deltas, and per-call device retention, for cases in the memory
  pass;
- `reason` — for anything not measured, why.

### Comparing a future branch

```bash
# on the optimization branch
python -m benchmarks.perfmap --output after.json --memory
# join after.csv against perfmap.csv on (name, backend)
```

The `name` and `backend` pair is stable across runs, and `tags.ladder` lets a
comparison ask the sharper question — did the *layer* that owned the cost get
cheaper, or did the whole stack shift?

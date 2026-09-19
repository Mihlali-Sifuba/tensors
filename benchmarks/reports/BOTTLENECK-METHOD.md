# How the bottleneck ranking was built

The ranking is derived from the ladder records, then filtered by judgment. Both
steps are written down here so the shortlist can be argued with.

## What a candidate is

One candidate is one **ladder**: the same computation measured at every layer
it could be isolated at, on the same dtype and size. That makes the two
overhead figures come from the same observation rather than from two
unrelated measurements:

- **absolute overhead** = outermost measured layer − innermost measured layer;
- **multiplicative overhead** = outermost ÷ innermost;
- **responsible step** = the adjacent layer pair with the largest added time.

## Rankings produced automatically

| Ranking | Sorted by | Filter |
| --- | --- | --- |
| largest absolute runtime | outermost layer time | ≥ 1,000 elements |
| largest absolute overhead | outermost − innermost | ≥ 1,000 elements |
| largest overhead ratio | outermost ÷ innermost | ≥ 1,000 elements |
| largest fixed overhead | outermost − innermost | ≤ 10 elements |

The ≥ 1,000-element filter exists so a ratio measured on a one-element
operation cannot lead a table. Fixed overhead is ranked separately, on its own
terms, because a per-call floor is a real cost but a different kind of one.

Two further rankings come from separate tables rather than from ladders:

- **largest CUDA synchronization cost** — from the barrier probe: host time
  minus device time, restricted to calls the probe found to block;
- **largest memory cost** — from the separate allocation pass: peak bytes per
  call, and per-call retention across a batch.

## Judgment applied on top

A ranking says which measurement is largest. It does not say which finding is
worth acting on. Three filters were applied by hand, and each is stated with
its reasoning in the report:

**1. Reachability.** A path that no public API can reach affects nothing today,
however large its ratio. The clearest case is the non-compact-layout gather:
its ratio is enormous, but no public operation returns a non-contiguous tensor,
so nothing in ordinary use pays it. It is reported as a **latent** finding —
one that becomes real the moment views are introduced — and not ranked with
the live ones.

**2. Breadth.** A cost inside a helper that every operation calls matters more
than the same cost in one operation. Breadth is estimated by counting the
measured cases that traverse the responsible code, and reported as a case
count rather than as a vague label.

**3. Realistic workload impact.** A microbenchmark ratio is only interesting
if it survives into a training step. The training suite measures forward, loss,
backward and optimizer phases separately for five model shapes and four batch
sizes, so every candidate is checked against them: does this path appear in a
training step, and how much of one does it account for? A candidate that does
not survive that check is reported as **microbenchmark-only**.

## Confidence

Each candidate carries a confidence, assigned from the measurement rather than
from impression:

| Confidence | Criteria |
| --- | --- |
| high | MAD ≤ 15% of median at every rung, effect ≥ 3×, and the mechanism is visible in the source and confirmed by a profiling probe |
| medium | one of: a rung flagged noisy, effect between 1.5× and 3×, or the mechanism inferred rather than confirmed |
| low | multiple rungs noisy, or the effect within the run's variance |

Where the profiling probes independently confirm a mechanism — an exact count
of provider calls, or an exact count of device-to-host conversions — that is
stated, because a counted mechanism is stronger evidence than a timing
difference.

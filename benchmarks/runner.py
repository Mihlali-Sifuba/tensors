"""The interleaving measurement scheduler.

A measurement here is a ``(case, backend)`` job. Jobs from every backend a
group supports are built together and then measured in rotated, seeded random
order across several rounds, so a thermal or allocator drift during the run
spreads across all of them instead of landing on whichever backend happened
to be measured last.
"""

from __future__ import annotations

import gc
import random
import statistics
import time
import traceback
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

import tensors as ts

from .case import Case, Classification, Group, Unsupported
from .measurement import SYNC_POLICY, Timing, timer_for
from .memory import measure_memory, release_device_memory
from .statistics import NOISE_THRESHOLD_PERCENT, summarize


@dataclass
class Job:
    """One case bound to the backend it will be measured on."""

    case: Case
    backend: str
    suite: str
    group: str
    loops: int = 1
    submit_samples: list[float] = field(default_factory=list)
    total_samples: list[float] = field(default_factory=list)
    device_samples: list[float] = field(default_factory=list)
    single_submit_samples: list[float] = field(default_factory=list)
    single_total_samples: list[float] = field(default_factory=list)
    classification: Classification = "measured"
    reason: str | None = None
    memory: dict[str, Any] | None = None
    synchronization: dict[str, Any] | None = None

    @property
    def key(self) -> str:
        """Return the identity used in reports and comparisons."""
        return f"{self.backend}::{self.case.name}"


def _calibrate(
    run: Callable[[], Any],
    timer: Any,
    target_seconds: float,
    *,
    single_shot: bool,
) -> int:
    """Choose an invocation count whose batch approaches the target duration."""
    if single_shot:
        return 1
    loops = 1
    ceiling = 5_000_000
    while loops < ceiling:
        elapsed = timer.run_batch(run, loops).total_seconds * loops
        if elapsed >= target_seconds:
            return loops
        if elapsed <= 0.0:
            scale = 16
        else:
            scale = max(2, min(16, int(target_seconds / elapsed) + 1))
        loops = min(loops * scale, ceiling)
    return loops


class Runner:
    """Builds, calibrates, and measures jobs group by group."""

    def __init__(
        self,
        *,
        backends: Sequence[str],
        rounds: int,
        target_seconds: float,
        seed: int,
        collect_memory: bool,
        progress: bool = True,
    ) -> None:
        self.backends = tuple(backends)
        self.rounds = rounds
        self.target_seconds = target_seconds
        self.random = random.Random(seed)
        self.seed = seed
        self.collect_memory = collect_memory
        self.progress = progress
        self.jobs: list[Job] = []
        self._started = time.perf_counter()

    # -- construction ---------------------------------------------------

    def _classified(
        self,
        group: Group,
        backend: str,
        classification: Classification,
        reason: str,
    ) -> Job:
        """Return a job that records why a group produced no measurement."""
        return Job(
            case=Case(
                name=group.name,
                run=lambda: None,
                family=group.suite,
            ),
            backend=backend,
            suite=group.suite,
            group=group.name,
            classification=classification,
            reason=reason,
        )

    def _build(self, group: Group) -> list[Job]:
        """Build every backend's jobs for one group, classifying failures."""
        jobs: list[Job] = []
        for backend in self.backends:
            try:
                with ts.use_backend(backend):
                    cases = list(group.factory(backend))
            except Unsupported as error:
                jobs.append(self._classified(group, backend, "unsupported", str(error)))
                continue
            except Exception as error:  # noqa: BLE001 - recorded, not hidden
                jobs.append(
                    self._classified(
                        group,
                        backend,
                        "error",
                        f"{type(error).__name__}: {error}\n"
                        + traceback.format_exc(limit=4),
                    )
                )
                continue
            for case in cases:
                if not case.supports(backend):
                    jobs.append(
                        Job(
                            case=case,
                            backend=backend,
                            suite=group.suite,
                            group=group.name,
                            classification="unsupported",
                            reason=(
                                "case declares this backend out of scope; "
                                f"eligible={sorted(case.backends or ())}"
                            ),
                        )
                    )
                    continue
                jobs.append(
                    Job(
                        case=case,
                        backend=backend,
                        suite=group.suite,
                        group=group.name,
                    )
                )
        return jobs

    # -- measurement ----------------------------------------------------

    def _prepare(self, job: Job, timer: Any) -> bool:
        """Validate and calibrate one job; return whether it can be measured."""
        case = job.case
        try:
            if case.setup is not None:
                case.setup()
            if case.reset is not None:
                case.reset()
            if case.validate is not None:
                case.validate()
            timer.prepare()
            if case.reset is not None:
                case.reset()
            job.loops = _calibrate(
                case.run,
                timer,
                self.target_seconds,
                single_shot=case.single_shot,
            )
            if case.reset is not None:
                case.reset()
        except Unsupported as error:
            job.classification = "unsupported"
            job.reason = str(error)
            return False
        except Exception as error:  # noqa: BLE001 - recorded, not hidden
            job.classification = "error"
            job.reason = f"{type(error).__name__}: {error}\n" + traceback.format_exc(
                limit=6
            )
            return False
        return True

    def _sample(self, job: Job, timer: Any) -> None:
        """Take one timed sample, warming the kernel cache first."""
        case = job.case
        if case.gc_enabled:
            gc.enable()
        else:
            gc.collect()
            gc.disable()
        try:
            # Entering a backend context clears the kernel-lookup cache, so
            # one untimed call restores steady state before sampling.
            if case.reset is not None:
                case.reset()
            case.run()
            if case.reset is not None:
                case.reset()
            timer.prepare()
            timing: Timing = timer.run_batch(case.run, job.loops)
        finally:
            gc.enable()
        job.submit_samples.append(timing.submit_seconds)
        job.total_samples.append(timing.total_seconds)
        if timing.device_seconds is not None:
            job.device_samples.append(timing.device_seconds)
        if timing.single_submit_seconds is not None:
            job.single_submit_samples.append(timing.single_submit_seconds)
        if timing.single_total_seconds is not None:
            job.single_total_samples.append(timing.single_total_seconds)

    def _memory_pass(self, job: Job) -> None:
        """Run the separate, untimed allocation pass for one job."""
        case = job.case
        try:
            if case.reset is not None:
                case.reset()
            sample = measure_memory(case.run, backend=job.backend)
        except Exception as error:  # noqa: BLE001 - recorded, not hidden
            job.memory = {"error": f"{type(error).__name__}: {error}"}
            return
        job.memory = sample.as_dict()

    def run(self, groups: Iterable[Group]) -> list[Job]:
        """Measure every group and return the completed jobs."""
        for group in groups:
            jobs = self._build(group)
            pending = [job for job in jobs if job.classification == "measured"]
            if self.progress and jobs:
                elapsed = time.perf_counter() - self._started
                print(
                    f"[{elapsed:7.1f}s] {group.name} ({len(pending)} jobs)",
                    flush=True,
                )
            try:
                self._measure_group(pending)
            finally:
                for job in jobs:
                    teardown = job.case.teardown
                    if teardown is not None:
                        try:
                            teardown()
                        except Exception:  # noqa: BLE001, S110
                            pass
                self.jobs.extend(jobs)
                # Inputs for this group die with the closures they were
                # captured in, so the next group starts from a clean pool.
                del jobs
                del pending
                gc.collect()
                release_device_memory()
        return self.jobs

    def _measure_group(self, pending: list[Job]) -> None:
        """Calibrate then interleave the measurable jobs of one group."""
        timers: dict[str, Any] = {}
        ready: list[Job] = []
        for job in pending:
            timer = timers.setdefault(job.backend, timer_for(job.backend))
            with ts.use_backend(job.backend):
                if self._prepare(job, timer):
                    ready.append(job)

        # Whether a call blocks is a property of the implementation, not of
        # the sample, so it is probed once per job rather than per round.
        for job in ready:
            timer = timers[job.backend]
            if not hasattr(timer, "probe_synchronization"):
                continue
            with ts.use_backend(job.backend):
                try:
                    if job.case.reset is not None:
                        job.case.reset()
                    job.synchronization = timer.probe_synchronization(job.case.run)
                    if job.case.reset is not None:
                        job.case.reset()
                except Exception as error:  # noqa: BLE001 - recorded
                    job.synchronization = {"error": f"{type(error).__name__}: {error}"}

        for round_index in range(self.rounds):
            order = list(ready)
            # Rotate first so comparable backends do not keep the same
            # relative position, then shuffle within the round.
            if order:
                offset = round_index % len(order)
                order = order[offset:] + order[:offset]
            self.random.shuffle(order)
            for job in order:
                with ts.use_backend(job.backend):
                    try:
                        self._sample(job, timers[job.backend])
                    except Exception as error:  # noqa: BLE001
                        job.classification = "error"
                        job.reason = (
                            f"{type(error).__name__}: {error}\n"
                            + traceback.format_exc(limit=6)
                        )

        if not self.collect_memory:
            return
        for job in ready:
            if not job.case.memory or job.classification != "measured":
                continue
            with ts.use_backend(job.backend):
                self._memory_pass(job)

    # -- reporting ------------------------------------------------------

    def settings(self) -> dict[str, Any]:
        """Return the methodology settings used for this run."""
        return {
            "backends": list(self.backends),
            "rounds": self.rounds,
            "samples_per_round": 1,
            "sample_count": self.rounds,
            "target_seconds_per_sample": self.target_seconds,
            "random_seed": self.seed,
            "ordering": (
                "jobs grouped by comparable workload; per round the group's "
                "jobs are rotated then shuffled with a seeded generator"
            ),
            "warmup": (
                "validation, calibration, and one untimed call per sample "
                "after entering the backend context"
            ),
            "synchronization_policy": SYNC_POLICY,
            "garbage_collection": (
                "disabled during timing except for cases that build cyclic "
                "graph objects, which keep collection in scope"
            ),
            "memory_pass": (
                "separate untimed tracemalloc and CuPy-pool pass"
                if self.collect_memory
                else "disabled"
            ),
            "noise_threshold_percent": NOISE_THRESHOLD_PERCENT,
            "statistic_for_comparison": "median of total_seconds",
        }


def job_record(job: Job) -> dict[str, Any]:
    """Return the serializable record for one measured or classified job."""
    case = job.case
    record: dict[str, Any] = {
        "name": case.name,
        "backend": job.backend,
        "suite": job.suite,
        "group": job.group,
        "layer": case.layer,
        "family": case.family,
        "dtype": case.dtype,
        "shape": (list(case.shape) if isinstance(case.shape, tuple) else case.shape),
        "elements": case.elements,
        "description": case.description,
        "tags": dict(case.tags),
        "classification": job.classification,
        "reason": job.reason,
    }
    if job.classification != "measured" or not job.total_samples:
        if job.classification == "measured":
            record["classification"] = "skipped"
            record["reason"] = record["reason"] or "no samples recorded"
        return record

    record["loops_per_sample"] = job.loops
    record["host_total"] = summarize(job.total_samples)
    record["host_submit"] = summarize(job.submit_samples)
    if job.device_samples:
        record["device"] = summarize(job.device_samples)
        submit_median = record["host_submit"]["median_seconds"]
        total_median = record["host_total"]["median_seconds"]
        device_median = record["device"]["median_seconds"]
        single_submit = (
            statistics.median(job.single_submit_samples)
            if job.single_submit_samples
            else None
        )
        single_total = (
            statistics.median(job.single_total_samples)
            if job.single_total_samples
            else None
        )
        probe = job.synchronization or {}
        record["cuda"] = {
            "submit_fraction_of_total": (
                None if total_median == 0 else submit_median / total_median
            ),
            "device_fraction_of_total": (
                None if total_median == 0 else device_median / total_median
            ),
            "host_overhead_seconds": total_median - device_median,
            "single_call_submit_seconds": single_submit,
            "single_call_total_seconds": single_total,
            "synchronization_probe": probe,
            # Only the barrier probe can tell a blocking call from one that
            # is merely expensive to launch.
            "hidden_synchronization": bool(probe.get("synchronizes")),
            "absorbed_fraction_of_barrier": probe.get("absorbed_fraction_of_barrier"),
        }
    median = record["host_total"]["median_seconds"]
    if case.work_items is not None:
        record["work_items"] = case.work_items
        record["work_items_per_second"] = (
            None if median == 0 else case.work_items / median
        )
    if job.memory is not None:
        record["memory"] = job.memory
    return record


__all__ = ["Job", "Runner", "job_record"]

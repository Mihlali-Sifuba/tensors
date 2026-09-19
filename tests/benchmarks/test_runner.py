"""The measurement contract every benchmark case relies on.

These pin behaviour a suite author depends on but cannot see: that a case is
set up before it is validated, torn down whatever happens, calibrated once,
and sampled once per round; that a backend a case declines is recorded as a
decision rather than dropped; and that a case which fails is reported with its
reason instead of ending the run.
"""

import unittest

import tensors as ts

from benchmarks.case import Case, Group, Unsupported
from benchmarks.runner import Runner, job_record
from benchmarks.statistics import NOISE_THRESHOLD_PERCENT, summarize
from benchmarks.measurement import SYNC_POLICY, CudaTimer, HostTimer, timer_for


def _runner(**overrides):
    settings = {
        "backends": ("python",),
        "rounds": 2,
        "target_seconds": 1e-6,
        "seed": 1,
        "collect_memory": False,
        "progress": False,
    }
    settings.update(overrides)
    return Runner(**settings)


def _group(factory, name="probe", suite="probe"):
    return Group(name=name, factory=factory, suite=suite)


class CaseLifecycleTests(unittest.TestCase):
    """Setup, validation, calibration, sampling, and teardown, in that order."""

    def test_a_case_is_set_up_first_and_torn_down_last(self):
        events: list[str] = []
        case = Case(
            name="lifecycle",
            run=lambda: events.append("run"),
            validate=lambda: events.append("validate"),
            setup=lambda: events.append("setup"),
            reset=lambda: events.append("reset"),
            teardown=lambda: events.append("teardown"),
            single_shot=True,
        )

        _runner().run([_group(lambda backend: [case])])

        self.assertEqual(events[0], "setup")
        self.assertEqual(events[-1], "teardown")
        self.assertEqual(events.count("validate"), 1)
        self.assertGreaterEqual(events.count("run"), 2)

    def test_a_failing_case_is_recorded_and_still_torn_down(self):
        """One broken case must not cost the rest of the run."""
        events: list[str] = []

        def fail() -> None:
            raise RuntimeError("invalid case")

        case = Case(
            name="invalid",
            run=lambda: None,
            validate=fail,
            teardown=lambda: events.append("teardown"),
        )

        jobs = _runner().run([_group(lambda backend: [case])])

        self.assertEqual(events, ["teardown"])
        self.assertEqual([job.classification for job in jobs], ["error"])
        self.assertIn("invalid case", jobs[0].reason)

    def test_a_single_shot_case_is_never_batched(self):
        """A state transition per sample cannot be repeated inside one."""
        case = Case(name="once", run=lambda: None, single_shot=True)

        jobs = _runner().run([_group(lambda backend: [case])])

        self.assertEqual(jobs[0].loops, 1)

    def test_samples_are_taken_once_per_round(self):
        case = Case(name="rounds", run=lambda: None, single_shot=True)

        jobs = _runner(rounds=4).run([_group(lambda backend: [case])])

        self.assertEqual(len(jobs[0].total_samples), 4)


class BackendEligibilityTests(unittest.TestCase):
    """A backend a case declines is a recorded decision, not an absence."""

    def test_a_case_can_restrict_its_backends(self):
        case = Case(name="numpy-only", run=lambda: None, backends=frozenset({"numpy"}))

        self.assertTrue(case.supports("numpy"))
        self.assertFalse(case.supports("python"))

    def test_a_declined_backend_produces_a_classified_job(self):
        case = Case(name="numpy-only", run=lambda: None, backends=frozenset({"numpy"}))

        jobs = _runner().run([_group(lambda backend: [case])])

        self.assertEqual([job.classification for job in jobs], ["unsupported"])
        self.assertIn("out of scope", jobs[0].reason)

    def test_a_factory_may_decline_a_whole_group(self):
        def factory(backend: str):
            raise Unsupported("nothing to measure here")

        jobs = _runner().run([_group(factory)])

        self.assertEqual([job.classification for job in jobs], ["unsupported"])
        self.assertEqual(jobs[0].reason, "nothing to measure here")

    def test_a_factory_that_raises_is_reported_not_swallowed(self):
        def factory(backend: str):
            raise ValueError("bad factory")

        jobs = _runner().run([_group(factory)])

        self.assertEqual([job.classification for job in jobs], ["error"])
        self.assertIn("bad factory", jobs[0].reason)


class StatisticsTests(unittest.TestCase):
    """Robust statistics, and a noise verdict rather than a quiet average."""

    def test_summary_reports_robust_and_classical_statistics(self):
        summary = summarize([1.0, 2.0, 3.0])

        self.assertEqual(summary["median_seconds"], 2.0)
        self.assertEqual(summary["min_seconds"], 1.0)
        self.assertEqual(summary["max_seconds"], 3.0)
        self.assertEqual(summary["sample_count"], 3)
        self.assertEqual(summary["samples_seconds"], [1.0, 2.0, 3.0])

    def test_raw_samples_are_retained_so_a_report_can_be_rebuilt(self):
        samples = [0.5, 0.75, 0.25]

        self.assertEqual(summarize(samples)["samples_seconds"], samples)

    def test_a_spread_beyond_the_threshold_is_called_noisy(self):
        self.assertFalse(summarize([1.0, 1.0, 1.0])["noisy"])
        self.assertTrue(summarize([1.0, 2.0, 3.0])["noisy"])
        self.assertGreater(NOISE_THRESHOLD_PERCENT, 0.0)

    def test_one_outlier_does_not_make_a_run_noisy(self):
        """The deviation is a median, so a single stall cannot dominate it."""
        self.assertFalse(summarize([1.0, 1.0, 10.0])["noisy"])

    def test_an_empty_sample_set_summarizes_to_nothing(self):
        self.assertEqual(summarize([]), {})


class RecordTests(unittest.TestCase):
    """What a measured job contributes to the report."""

    def test_a_record_carries_the_layer_and_family_it_declared(self):
        case = Case(
            name="labelled",
            run=lambda: None,
            layer="dispatch",
            family="probe/family",
            dtype="float32",
            shape=(2, 3),
            elements=6,
            single_shot=True,
        )

        jobs = _runner().run([_group(lambda backend: [case])])
        record = job_record(jobs[0])

        self.assertEqual(record["layer"], "dispatch")
        self.assertEqual(record["family"], "probe/family")
        self.assertEqual(record["dtype"], "float32")
        self.assertEqual(record["shape"], [2, 3])
        self.assertEqual(record["elements"], 6)
        self.assertEqual(record["classification"], "measured")

    def test_a_record_keeps_its_raw_samples(self):
        case = Case(name="samples", run=lambda: None, single_shot=True)

        jobs = _runner(rounds=3).run([_group(lambda backend: [case])])
        record = job_record(jobs[0])

        self.assertEqual(len(record["host_total"]["samples_seconds"]), 3)

    def test_a_classified_job_records_its_reason_and_no_timing(self):
        def factory(backend: str):
            raise Unsupported("declined")

        jobs = _runner().run([_group(factory)])
        record = job_record(jobs[0])

        self.assertEqual(record["classification"], "unsupported")
        self.assertEqual(record["reason"], "declined")
        self.assertNotIn("host_total", record)


class TimerSelectionTests(unittest.TestCase):
    """One synchronization policy, chosen by the backend's execution model."""

    def test_a_host_backend_gets_the_host_timer(self):
        self.assertIsInstance(timer_for("python"), HostTimer)
        self.assertIsInstance(timer_for("numpy"), HostTimer)

    @unittest.skipUnless(
        "cuda" in ts.available_backends(), "CUDA backend is not installed"
    )
    def test_cuda_gets_the_event_recording_timer(self):
        self.assertIsInstance(timer_for("cuda"), CudaTimer)

    def test_a_host_timing_reports_no_device_time(self):
        timing = HostTimer().run_batch(lambda: None, 4)

        self.assertIsNone(timing.device_seconds)
        self.assertEqual(timing.submit_seconds, timing.total_seconds)

    def test_the_synchronization_policy_is_stated(self):
        """The policy is part of the result, so it has to be legible."""
        self.assertIn("no synchronization between measured", SYNC_POLICY)
        self.assertIn("batch boundary", SYNC_POLICY)


class RunnerSettingsTests(unittest.TestCase):
    """The methodology a stored report has to carry."""

    def test_settings_describe_how_the_run_was_taken(self):
        settings = _runner(rounds=5, target_seconds=0.05).settings()

        self.assertEqual(settings["backends"], ["python"])
        self.assertEqual(settings["rounds"], 5)
        self.assertEqual(settings["sample_count"], 5)
        self.assertEqual(settings["target_seconds_per_sample"], 0.05)
        self.assertEqual(settings["random_seed"], 1)
        self.assertIn("rotated then shuffled", settings["ordering"])


if __name__ == "__main__":
    unittest.main()

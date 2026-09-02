"""Tests for benchmark report construction and presentation."""

from __future__ import annotations

import io
import unittest
from contextlib import redirect_stdout
from typing import Any
from unittest.mock import patch

import tensors as ts
from benchmarks.__main__ import _cases, _parser
from benchmarks.runner import (
    BenchmarkCase,
    combine_backend_reports,
    measure,
    print_backend_comparison,
    print_report,
)


def _report(backend: str, seconds: float) -> dict[str, Any]:
    return {
        "metadata": {"backend": backend},
        "settings": {"repeats": 3, "target_seconds_per_sample": 0.02},
        "benchmarks": {
            "tensor.add/100": {
                "median_seconds": seconds,
                "variability_percent": 1.25,
                "work_items_per_second": 100 / seconds,
            },
        },
    }


class BackendBenchmarkTests(unittest.TestCase):
    def test_all_available_backends_are_the_cli_default(self) -> None:
        self.assertEqual(_parser().parse_args([]).backend, "all")

    def test_compact_core_suite_is_the_cli_default(self) -> None:
        self.assertEqual(_parser().parse_args([]).suite, "core")

    def test_accelerated_backend_selector_is_available(self) -> None:
        arguments = _parser().parse_args([
            "--backend",
            "accelerated",
            "--suite",
            "storage",
        ])

        self.assertEqual(arguments.backend, "accelerated")
        self.assertEqual(arguments.suite, "storage")

    def test_case_can_restrict_eligible_backends(self) -> None:
        case = BenchmarkCase(
            name="provider.add/100",
            run=lambda: None,
            backends=frozenset({"numpy", "cuda"}),
        )

        self.assertFalse(case.supports_backend("python"))
        self.assertTrue(case.supports_backend("numpy"))
        self.assertTrue(case.supports_backend("cuda"))

    def test_measurement_records_the_benchmark_layer(self) -> None:
        result = measure(
            BenchmarkCase(
                name="kernel.add/1",
                run=lambda: 2,
                validate=lambda: None,
                layer="kernel",
            ),
            repeats=1,
            target_seconds=1e-9,
        )

        self.assertEqual(result["layer"], "kernel")

    def test_combined_report_preserves_each_backend(self) -> None:
        reports = {
            "python": _report("python", 0.004),
            "numpy": _report("numpy", 0.001),
        }

        combined = combine_backend_reports(reports)

        self.assertEqual(combined["metadata"]["backends"], ["python", "numpy"])
        self.assertIs(combined["backends"]["python"], reports["python"])
        self.assertIs(combined["backends"]["numpy"], reports["numpy"])

    def test_comparison_prints_backend_medians_and_speedup(self) -> None:
        combined = combine_backend_reports({
            "python": _report("python", 0.004),
            "numpy": _report("numpy", 0.001),
            "cuda": _report("cuda", 0.0005),
        })
        output = io.StringIO()

        with redirect_stdout(output):
            print_backend_comparison(combined)

        rendered = output.getvalue()
        self.assertIn("python median", rendered)
        self.assertIn("numpy median", rendered)
        self.assertIn("cuda median", rendered)
        self.assertIn("4.00x", rendered)
        self.assertIn("8.00x", rendered)

    def test_comparison_supports_backend_specific_cases(self) -> None:
        python_report = _report("python", 0.004)
        numpy_report = _report("numpy", 0.001)
        numpy_report["benchmarks"]["provider.add/100"] = {
            "median_seconds": 0.0001,
            "variability_percent": 0.5,
        }
        combined = combine_backend_reports({
            "python": python_report,
            "numpy": numpy_report,
        })
        output = io.StringIO()

        with redirect_stdout(output):
            print_backend_comparison(combined)

        provider_row = next(
            line
            for line in output.getvalue().splitlines()
            if line.startswith("provider.add/100")
        )
        self.assertIn("-", provider_row)
        self.assertIn("100.00 us", provider_row)

    def test_single_backend_report_identifies_backend(self) -> None:
        output = io.StringIO()

        with redirect_stdout(output):
            print_report(_report("python", 0.004))

        self.assertIn("Backend: python", output.getvalue())

    def test_measure_resets_samples_and_always_tears_down(self) -> None:
        events: list[str] = []
        case = BenchmarkCase(
            name="lifecycle",
            run=lambda: events.append("run"),
            validate=lambda: events.append("validate"),
            setup=lambda: events.append("setup"),
            reset=lambda: events.append("reset"),
            teardown=lambda: events.append("teardown"),
        )

        measure(case, repeats=1, target_seconds=1e-9)

        self.assertEqual(events[0], "setup")
        self.assertEqual(events[-1], "teardown")
        self.assertEqual(events.count("validate"), 1)
        self.assertEqual(events.count("reset"), 2)

    def test_measure_tears_down_after_validation_failure(self) -> None:
        events: list[str] = []

        def fail() -> None:
            raise RuntimeError("invalid case")

        case = BenchmarkCase(
            name="invalid",
            run=lambda: None,
            validate=fail,
            teardown=lambda: events.append("teardown"),
        )

        with self.assertRaisesRegex(RuntimeError, "invalid case"):
            measure(case, repeats=1, target_seconds=1e-9)

        self.assertEqual(events, ["teardown"])

    def test_all_suite_match_skips_unrelated_factories(self) -> None:
        calls: list[str] = []

        def autograd_factory() -> list[BenchmarkCase]:
            calls.append("autograd")
            return [BenchmarkCase("autograd.grad/1", lambda: None)]

        def training_factory() -> list[BenchmarkCase]:
            calls.append("training")
            return [BenchmarkCase("training.step/1", lambda: None)]

        with (
            patch(
                "benchmarks.__main__.SUITES",
                {"autograd": autograd_factory, "training": training_factory},
            ),
            patch(
                "benchmarks.__main__.SUITE_PREFIXES",
                {
                    "autograd": frozenset({"autograd"}),
                    "training": frozenset({"training"}),
                },
            ),
        ):
            selected = _cases("all", "autograd.grad")

        self.assertEqual([case.name for case in selected], ["autograd.grad/1"])
        self.assertEqual(calls, ["autograd"])

    def test_every_python_benchmark_case_validates(self) -> None:
        with ts.use_backend("python"):
            cases = [
                case for case in _cases("all")
                if case.supports_backend("python")
            ]
            for case in cases:
                with self.subTest(case=case.name):
                    if case.setup is not None:
                        case.setup()
                    try:
                        if case.reset is not None:
                            case.reset()
                        if case.validate is not None:
                            case.validate()
                    finally:
                        if case.teardown is not None:
                            case.teardown()


if __name__ == "__main__":
    unittest.main()

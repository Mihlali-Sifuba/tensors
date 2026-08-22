"""Tests for benchmark report construction and presentation."""

from __future__ import annotations

import io
import unittest
from contextlib import redirect_stdout
from typing import Any

from benchmarks.__main__ import _parser
from benchmarks.runner import (
    combine_backend_reports,
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
        })
        output = io.StringIO()

        with redirect_stdout(output):
            print_backend_comparison(combined)

        rendered = output.getvalue()
        self.assertIn("python median", rendered)
        self.assertIn("numpy median", rendered)
        self.assertIn("4.00x", rendered)

    def test_single_backend_report_identifies_backend(self) -> None:
        output = io.StringIO()

        with redirect_stdout(output):
            print_report(_report("python", 0.004))

        self.assertIn("Backend: python", output.getvalue())


if __name__ == "__main__":
    unittest.main()

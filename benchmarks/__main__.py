"""Command-line entry point for the benchmark package."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path
from typing import Any

import tensors as ts

from . import (
    autograd_cases,
    backend_cases,
    graph_cases,
    tensor_cases,
    training_cases,
)
from .runner import (
    BenchmarkCase,
    combine_backend_reports,
    print_backend_comparison,
    print_report,
    run_suite,
    write_report,
)


CaseFactory = Callable[[], list[BenchmarkCase]]
SUITES: dict[str, CaseFactory] = {
    "tensor": tensor_cases.cases,
    "backend": backend_cases.cases,
    "graph": graph_cases.cases,
    "autograd": autograd_cases.cases,
    "training": training_cases.cases,
}


def _cases(suite: str) -> list[BenchmarkCase]:
    if suite == "all":
        return [
            case
            for factory in SUITES.values()
            for case in factory()
        ]
    return SUITES[suite]()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark the public tensors API.",
    )
    parser.add_argument(
        "--backend",
        choices=("all", "python", "numpy", "auto"),
        default="all",
        help="backend to benchmark (default: all available backends)",
    )
    parser.add_argument(
        "--suite",
        choices=("all", *SUITES),
        default="all",
        help="benchmark group to run (default: all)",
    )
    parser.add_argument(
        "--match",
        metavar="TEXT",
        help="run only case names containing this text",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="use three short samples for a fast smoke run",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        help="number of measured samples per case",
    )
    parser.add_argument(
        "--target-time",
        type=float,
        help="calibration target in seconds per sample",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="optional JSON result path",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="list selected cases without running them",
    )
    return parser


def main() -> int:
    parser = _parser()
    arguments = parser.parse_args()
    if arguments.backend == "all":
        backends = ts.available_backends()
    else:
        try:
            with ts.use_backend(arguments.backend):
                backends = (ts.get_backend(),)
        except (ValueError, ts.BackendUnavailableError) as error:
            parser.error(str(error))
    selected = _cases(arguments.suite)
    if arguments.match:
        selected = [
            case for case in selected
            if arguments.match.casefold() in case.name.casefold()
        ]
    if not selected:
        parser.error("no benchmark cases matched")

    if arguments.list:
        for case in selected:
            print(case.name)
        return 0

    repeats = arguments.repeats
    if repeats is None:
        repeats = 3 if arguments.quick else 7
    if repeats <= 0:
        parser.error("--repeats must be positive")

    target_seconds = arguments.target_time
    if target_seconds is None:
        target_seconds = 0.02 if arguments.quick else 0.2
    if target_seconds <= 0.0:
        parser.error("--target-time must be positive")

    reports: dict[str, dict[str, Any]] = {}
    for backend in backends:
        with ts.use_backend(backend):
            backend_cases = _cases(arguments.suite)
            if arguments.match:
                backend_cases = [
                    case for case in backend_cases
                    if arguments.match.casefold() in case.name.casefold()
                ]
            reports[backend] = run_suite(
                backend_cases,
                repeats=repeats,
                target_seconds=target_seconds,
            )

    if len(reports) == 1:
        report = next(iter(reports.values()))
        print_report(report)
    else:
        report = combine_backend_reports(reports)
        print_backend_comparison(report)
    if arguments.output is not None:
        write_report(report, arguments.output)
        print(f"\nWrote {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

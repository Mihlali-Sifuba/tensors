"""Command-line entry point for the benchmark suite."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import tensors as ts

from . import registry
from .harness import Runner, job_record
from .memory import device_memory_status
from .meta import environment_metadata
from .report import (
    build_report,
    print_summary,
    write_csv,
    write_json,
    write_samples_csv,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m benchmarks",
        description=(
            "Measure the tensors package at every layer of its execution "
            "stack and write a machine-readable performance map."
        ),
    )
    parser.add_argument(
        "--suite",
        action="append",
        metavar="NAME",
        help=(
            "suite to run; repeatable. Defaults to every suite. "
            "Use --list-suites to see the names."
        ),
    )
    parser.add_argument(
        "--backend",
        action="append",
        choices=("python", "numpy", "cuda"),
        metavar="NAME",
        help="backend to measure; repeatable. Defaults to all installed.",
    )
    parser.add_argument(
        "--match",
        metavar="TEXT",
        help="run only groups whose name contains this text",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=5,
        help="measured samples per case, one per interleaved round",
    )
    parser.add_argument(
        "--target-time",
        type=float,
        default=0.05,
        help="calibration target in seconds per sample",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260910,
        help="seed for the per-round execution order",
    )
    parser.add_argument(
        "--memory",
        action="store_true",
        help="run the separate allocation pass for cases that ask for it",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="three short rounds for verifying the harness",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="JSON result path; CSV siblings are written beside it",
    )
    parser.add_argument(
        "--list-suites",
        action="store_true",
        help="list suite names and exit",
    )
    parser.add_argument(
        "--list-groups",
        action="store_true",
        help="list the selected groups without measuring them",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    arguments = parser.parse_args(argv)

    if arguments.list_suites:
        for name in registry.SUITE_MODULES:
            print(name)
        return 0

    suites = arguments.suite or list(registry.DEFAULT_SUITES)
    unknown = [name for name in suites if name not in registry.SUITE_MODULES]
    if unknown:
        parser.error(f"unknown suite(s): {', '.join(unknown)}")

    backends = arguments.backend or list(ts.available_backends())
    missing = [
        name for name in backends if name not in ts.available_backends()
    ]
    if missing:
        parser.error(
            f"backend(s) not installed: {', '.join(missing)}; available: "
            f"{', '.join(ts.available_backends())}"
        )

    groups = registry.collect(suites, match=arguments.match)
    if not groups:
        parser.error("no groups matched the selection")

    if arguments.list_groups:
        for group in groups:
            print(f"{group.suite:14} {group.name}")
        print(f"\n{len(groups)} groups")
        return 0

    rounds = 3 if arguments.quick else arguments.rounds
    target = 0.01 if arguments.quick else arguments.target_time
    if rounds <= 0:
        parser.error("--rounds must be positive")
    if target <= 0.0:
        parser.error("--target-time must be positive")

    print(
        f"Measuring {len(groups)} groups on "
        f"{', '.join(backends)} with {rounds} rounds "
        f"(target {target * 1000:.0f} ms/sample)",
        flush=True,
    )
    started = time.perf_counter()
    runner = Runner(
        backends=backends,
        rounds=rounds,
        target_seconds=target,
        seed=arguments.seed,
        collect_memory=arguments.memory,
    )
    jobs = runner.run(groups)
    elapsed = time.perf_counter() - started

    records = [job_record(job) for job in jobs]
    metadata = environment_metadata()
    metadata["run"] = {
        "suites": suites,
        "group_count": len(groups),
        "record_count": len(records),
        "wall_clock_seconds": elapsed,
        "match": arguments.match,
        "device_memory_after": device_memory_status(),
        "command": " ".join(sys.argv),
    }
    report = build_report(
        metadata=metadata,
        settings=runner.settings(),
        records=records,
    )
    print_summary(report)
    print(f"\nCompleted {len(records)} records in {elapsed:.1f}s")

    if arguments.output is not None:
        output = arguments.output
        write_json(report, output)
        write_csv(records, output.with_suffix(".csv"))
        write_samples_csv(
            records,
            output.with_name(output.stem + "-samples.csv"),
        )
        print(
            f"Wrote {output}, {output.with_suffix('.csv')}, and "
            f"{output.with_name(output.stem + '-samples.csv')}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""What the benchmark suite would measure, without measuring it.

Building a case is cheap; measuring it is not. Listing the built cases makes
the benchmark matrix something that can be inspected, diffed, and asserted
on: which operations are covered, at which layers, on which backends, over
which dtypes and shapes, and which combinations a backend declines.

This exists to keep a structural change structural. Moving a suite between
modules must not alter the matrix it produces, and an inventory taken before
and after a move says whether it did.

    python -m benchmarks.inventory --output before.json
    python -m benchmarks.inventory --summary
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import tensors as ts

from benchmarks import registry
from benchmarks.case import Case, Group, Unsupported


def _shape(value: Any) -> Any:
    """Render a case shape as something JSON can hold and sort."""
    if isinstance(value, tuple):
        return list(value)
    return value


def _row(
    case: Case,
    *,
    backend: str,
    suite: str,
    group: str,
    status: str,
    reason: str | None = None,
) -> dict[str, Any]:
    """Return one inventory row: what this case is, not how fast it is."""
    return {
        "name": case.name,
        "backend": backend,
        "suite": suite,
        "group": group,
        "layer": case.layer,
        "family": case.family,
        "dtype": case.dtype,
        "shape": _shape(case.shape),
        "elements": case.elements,
        "work_items": case.work_items,
        "tags": dict(case.tags),
        "memory": case.memory,
        "single_shot": case.single_shot,
        "status": status,
        "reason": reason,
    }


def _declined(group: Group, backend: str, status: str, reason: str) -> dict[str, Any]:
    """Return the row standing in for a group that built no cases."""
    return _row(
        Case(name=group.name, run=lambda: None, family=group.suite),
        backend=backend,
        suite=group.suite,
        group=group.name,
        status=status,
        reason=reason,
    )


def rows_for(group: Group, backends: Sequence[str]) -> Iterator[dict[str, Any]]:
    """Yield an inventory row for every case ``group`` builds, or why it did not.

    A factory that raises :class:`Unsupported` is stating a deliberate
    exclusion, and that statement is part of the matrix: it records that the
    combination was considered and declined, which is different from never
    having been offered.
    """
    for backend in backends:
        try:
            with ts.use_backend(backend):
                cases = list(group.factory(backend))
        except Unsupported as error:
            yield _declined(group, backend, "unsupported", str(error))
            continue
        except Exception as error:  # noqa: BLE001 - recorded, not hidden
            yield _declined(
                group,
                backend,
                "error",
                f"{type(error).__name__}: {error}\n{traceback.format_exc(limit=3)}",
            )
            continue
        for case in cases:
            if case.supports(backend):
                yield _row(
                    case,
                    backend=backend,
                    suite=group.suite,
                    group=group.name,
                    status="supported",
                )
            else:
                yield _row(
                    case,
                    backend=backend,
                    suite=group.suite,
                    group=group.name,
                    status="unsupported",
                    reason=(
                        "case declares this backend out of scope; "
                        f"eligible={sorted(case.backends or ())}"
                    ),
                )


def build(
    suites: Sequence[str] | None = None,
    backends: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """Return the inventory for the selected suites and backends."""
    names = tuple(suites) if suites else registry.DEFAULT_SUITES
    targets = tuple(backends) if backends else ts.available_backends()
    inventory: list[dict[str, Any]] = []
    for group in registry.collect(names):
        inventory.extend(rows_for(group, targets))
    inventory.sort(key=lambda row: (row["backend"], row["suite"], row["name"]))
    return inventory


def summarize(inventory: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Return counts that a person can read and a test can assert on."""
    by_suite: dict[str, dict[str, int]] = {}
    by_layer: dict[str, int] = {}
    by_status: dict[str, int] = {}
    for row in inventory:
        suite = by_suite.setdefault(row["suite"], {})
        suite[row["status"]] = suite.get(row["status"], 0) + 1
        by_layer[row["layer"]] = by_layer.get(row["layer"], 0) + 1
        by_status[row["status"]] = by_status.get(row["status"], 0) + 1
    return {
        "total": len(inventory),
        "by_status": dict(sorted(by_status.items())),
        "by_layer": dict(sorted(by_layer.items())),
        "by_suite": {
            name: dict(sorted(counts.items()))
            for name, counts in sorted(by_suite.items())
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m benchmarks.inventory",
        description="List the cases the benchmark suite would measure.",
    )
    parser.add_argument(
        "--suite",
        action="append",
        metavar="NAME",
        help="suite to inventory; repeatable. Defaults to every suite.",
    )
    parser.add_argument(
        "--backend",
        action="append",
        choices=("python", "numpy", "cuda"),
        metavar="NAME",
        help="backend to inventory; repeatable. Defaults to all installed.",
    )
    parser.add_argument("--output", type=Path, help="JSON inventory path")
    parser.add_argument(
        "--summary",
        action="store_true",
        help="print counts instead of the full inventory",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    inventory = build(arguments.suite, arguments.backend)
    totals = summarize(inventory)
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(
            json.dumps(inventory, indent=1, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"wrote {len(inventory)} rows to {arguments.output}")
    if arguments.summary or arguments.output is None:
        json.dump(totals, sys.stdout, indent=1)
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

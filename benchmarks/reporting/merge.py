"""Combine per-suite reports into one performance map.

Running the whole map in stages is safer than one long process, so the
stages are merged afterwards. The merge re-derives every analysis table
from the combined records, which is what makes cross-suite comparisons —
a ladder whose rungs came from different suites, for instance — resolve.

Usage::

    python -m benchmarks.merge out/*.json --output out/perfmap.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .console import print_summary
from .csv import write_csv, write_samples_csv
from .json import READABLE_SCHEMAS, build_report, schema_of, write_json


def _load(
    paths: list[Path],
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    """Return merged metadata, settings, and records from several reports."""
    records: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    metadata: dict[str, Any] = {}
    settings: dict[str, Any] = {}
    stages: list[dict[str, Any]] = []
    duplicates = 0

    schemas: list[str] = []
    for path in sorted(paths):
        report = json.loads(path.read_text(encoding="utf-8"))
        version = schema_of(report)
        if version not in READABLE_SCHEMAS:
            raise ValueError(
                f"{path.name} is schema {version}, which this package cannot "
                f"read; it understands {', '.join(READABLE_SCHEMAS)}"
            )
        schemas.append(version)
        if not metadata:
            metadata = dict(report["metadata"])
        if not settings:
            settings = dict(report["settings"])
        run = report["metadata"].get("run", {})
        stages.append(
            {
                "file": path.name,
                "suites": run.get("suites"),
                "record_count": run.get("record_count"),
                "wall_clock_seconds": run.get("wall_clock_seconds"),
                "timestamp_utc": report["metadata"].get("timestamp_utc"),
                "commit": report["metadata"].get("git", {}).get("commit"),
            }
        )
        for record in report["records"]:
            key = (record["backend"], record["name"])
            if key in seen:
                duplicates += 1
                continue
            seen.add(key)
            records.append(record)

    commits = {stage["commit"] for stage in stages if stage.get("commit")}
    metadata["run"] = {
        "stages": stages,
        "record_count": len(records),
        "duplicate_records_dropped": duplicates,
        "wall_clock_seconds": sum(
            stage["wall_clock_seconds"] or 0.0 for stage in stages
        ),
        "suites": sorted(
            {suite for stage in stages for suite in (stage["suites"] or [])}
        ),
        "commits": sorted(commits),
        "single_commit": len(commits) <= 1,
        # Suite labels mean different things in different schema versions, so
        # a merge across versions is recorded rather than smoothed over.
        "schemas": sorted(set(schemas)),
        "single_schema": len(set(schemas)) <= 1,
    }
    return metadata, settings, records


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m benchmarks.merge",
        description="Merge per-suite performance-map reports into one.",
    )
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args(argv)

    paths: list[Path] = []
    for item in arguments.inputs:
        paths.extend(
            sorted(item.parent.glob(item.name)) if "*" in item.name else [item]
        )
    missing = [path for path in paths if not path.is_file()]
    if missing:
        parser.error("missing input(s): " + ", ".join(str(path) for path in missing))

    metadata, settings, records = _load(paths)
    if not metadata["run"]["single_commit"]:
        print(
            "WARNING: inputs were measured at different commits: "
            + ", ".join(metadata["run"]["commits"])
        )
    report = build_report(metadata=metadata, settings=settings, records=records)
    print_summary(report)
    write_json(report, arguments.output)
    write_csv(records, arguments.output.with_suffix(".csv"))
    write_samples_csv(
        records,
        arguments.output.with_name(arguments.output.stem + "-samples.csv"),
    )
    print(
        f"\nMerged {len(paths)} reports into {len(records)} records "
        f"({metadata['run']['duplicate_records_dropped']} duplicates "
        f"dropped)"
    )
    print(f"Wrote {arguments.output} and its CSV siblings")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

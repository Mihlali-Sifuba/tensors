"""Serialization and console presentation of a performance-map run."""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from . import analysis


def duration(seconds: float | None) -> str:
    """Format a duration with a unit chosen for readability."""
    if seconds is None:
        return "-"
    if seconds != seconds:  # NaN
        return "nan"
    if seconds < 1e-6:
        return f"{seconds * 1e9:.1f}ns"
    if seconds < 1e-3:
        return f"{seconds * 1e6:.2f}us"
    if seconds < 1.0:
        return f"{seconds * 1e3:.2f}ms"
    return f"{seconds:.3f}s"


def ratio(value: float | None) -> str:
    """Format a multiplicative overhead."""
    if value is None:
        return "-"
    if value >= 100:
        return f"{value:.0f}x"
    if value >= 10:
        return f"{value:.1f}x"
    return f"{value:.2f}x"


def byte_size(value: float | None) -> str:
    """Format a byte count."""
    if value is None:
        return "-"
    magnitude = abs(value)
    sign = "-" if value < 0 else ""
    for unit, scale in (("GB", 1e9), ("MB", 1e6), ("KB", 1e3)):
        if magnitude >= scale:
            return f"{sign}{magnitude / scale:.2f}{unit}"
    return f"{sign}{magnitude:.0f}B"


def _finite(value: Any) -> Any:
    """Replace non-finite floats so the JSON stays strictly valid."""
    if isinstance(value, float):
        if math.isnan(value):
            return None
        if math.isinf(value):
            return "Infinity" if value > 0 else "-Infinity"
        return value
    if isinstance(value, dict):
        return {key: _finite(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite(item) for item in value]
    return value


def write_json(report: dict[str, Any], output: Path) -> None:
    """Write the complete machine-readable report."""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(_finite(report), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


#: The flat schema a comparison against a future branch is joined on.
CSV_COLUMNS: tuple[str, ...] = (
    "name",
    "backend",
    "suite",
    "group",
    "layer",
    "family",
    "dtype",
    "shape",
    "elements",
    "classification",
    "loops_per_sample",
    "sample_count",
    "median_seconds",
    "mean_seconds",
    "stdev_seconds",
    "min_seconds",
    "max_seconds",
    "p95_seconds",
    "mad_seconds",
    "variability_percent",
    "noisy",
    "submit_median_seconds",
    "device_median_seconds",
    "host_overhead_seconds",
    "single_call_submit_seconds",
    "single_call_total_seconds",
    "absorbed_fraction_of_barrier",
    "hidden_synchronization",
    "work_items",
    "work_items_per_second",
    "host_peak_bytes",
    "host_allocation_count",
    "host_retained_bytes_per_call",
    "device_used_delta_bytes",
    "device_retained_bytes_per_call",
    "ladder",
    "curve",
    "reason",
)


def _csv_row(record: dict[str, Any]) -> dict[str, Any]:
    total = record.get("host_total") or {}
    submit = record.get("host_submit") or {}
    device = record.get("device") or {}
    cuda = record.get("cuda") or {}
    memory = record.get("memory") or {}
    tags = record.get("tags") or {}
    shape = record.get("shape")
    return {
        "name": record.get("name"),
        "backend": record.get("backend"),
        "suite": record.get("suite"),
        "group": record.get("group"),
        "layer": record.get("layer"),
        "family": record.get("family"),
        "dtype": record.get("dtype"),
        "shape": (
            "x".join(str(item) for item in shape)
            if isinstance(shape, list)
            else shape
        ),
        "elements": record.get("elements"),
        "classification": record.get("classification"),
        "loops_per_sample": record.get("loops_per_sample"),
        "sample_count": total.get("sample_count"),
        "median_seconds": total.get("median_seconds"),
        "mean_seconds": total.get("mean_seconds"),
        "stdev_seconds": total.get("stdev_seconds"),
        "min_seconds": total.get("min_seconds"),
        "max_seconds": total.get("max_seconds"),
        "p95_seconds": total.get("p95_seconds"),
        "mad_seconds": total.get("median_absolute_deviation_seconds"),
        "variability_percent": total.get("variability_percent"),
        "noisy": total.get("noisy"),
        "submit_median_seconds": submit.get("median_seconds"),
        "device_median_seconds": device.get("median_seconds"),
        "host_overhead_seconds": cuda.get("host_overhead_seconds"),
        "single_call_submit_seconds": cuda.get("single_call_submit_seconds"),
        "single_call_total_seconds": cuda.get("single_call_total_seconds"),
        "absorbed_fraction_of_barrier": cuda.get(
            "absorbed_fraction_of_barrier"
        ),
        "hidden_synchronization": cuda.get("hidden_synchronization"),
        "work_items": record.get("work_items"),
        "work_items_per_second": record.get("work_items_per_second"),
        "host_peak_bytes": memory.get("host_peak_bytes"),
        "host_allocation_count": memory.get("host_allocation_count"),
        "host_retained_bytes_per_call": memory.get(
            "host_retained_bytes_per_call"
        ),
        "device_used_delta_bytes": memory.get("device_used_delta_bytes"),
        "device_retained_bytes_per_call": memory.get(
            "device_retained_bytes_per_call"
        ),
        "ladder": tags.get("ladder"),
        "curve": tags.get("curve"),
        "reason": (record.get("reason") or "").splitlines()[0]
        if record.get("reason")
        else None,
    }


def write_csv(records: Iterable[dict[str, Any]], output: Path) -> None:
    """Write one flat row per record for spreadsheet and diff comparison."""
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for record in records:
            writer.writerow(_csv_row(record))


def write_samples_csv(
    records: Iterable[dict[str, Any]],
    output: Path,
) -> None:
    """Write every retained raw sample, one row each."""
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["name", "backend", "layer", "elements", "metric",
             "sample_index", "seconds", "loops_per_sample"]
        )
        for record in records:
            if record.get("classification") != "measured":
                continue
            loops = record.get("loops_per_sample")
            for metric in ("host_total", "host_submit", "device"):
                block = record.get(metric)
                if not block:
                    continue
                for index, value in enumerate(block["samples_seconds"]):
                    writer.writerow([
                        record["name"],
                        record["backend"],
                        record["layer"],
                        record.get("elements"),
                        metric,
                        index,
                        value,
                        loops,
                    ])


# -- console tables -----------------------------------------------------


def _table(
    heading: Sequence[str],
    rows: Sequence[Sequence[str]],
    *,
    title: str,
) -> None:
    if not rows:
        return
    widths = [len(column) for column in heading]
    for row in rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))
    line = "  ".join(
        column.ljust(widths[index]) for index, column in enumerate(heading)
    )
    print(f"\n{title}")
    print(line)
    print("-" * len(line))
    for row in rows:
        print("  ".join(
            cell.ljust(widths[index]) for index, cell in enumerate(row)
        ))


def print_summary(report: dict[str, Any], *, limit: int = 20) -> None:
    """Print the tables that make a run readable without opening the JSON."""
    records = report["records"]
    derived = report["analysis"]
    counts = derived["classifications"]["counts"]
    print(
        "\nRecords: "
        + ", ".join(f"{key}={value}" for key, value in sorted(counts.items()))
    )

    _table(
        ("ladder", "backend", "dtype", "elems", "base", "top", "ratio",
         "responsible step"),
        [
            (
                item["ladder"],
                item["backend"],
                str(item["dtype"]),
                str(item["elements"]),
                duration(item["base_seconds"]),
                duration(item["top_seconds"]),
                ratio(item["multiplicative_overhead"]),
                str(item["responsible_layer"]),
            )
            for item in derived["bottlenecks"]["by_absolute_overhead"][:limit]
        ],
        title="Largest absolute library overhead (>=1000 elements)",
    )

    _table(
        ("ladder", "backend", "dtype", "elems", "ratio", "added"),
        [
            (
                item["ladder"],
                item["backend"],
                str(item["dtype"]),
                str(item["elements"]),
                ratio(item["multiplicative_overhead"]),
                duration(item["absolute_overhead_seconds"]),
            )
            for item in derived["bottlenecks"]["by_overhead_ratio"][:limit]
        ],
        title="Largest overhead ratio (>=1000 elements)",
    )

    _table(
        ("case", "elems", "host", "device", "host-device", "1-call",
         "absorbed", "blocks"),
        [
            (
                item["name"],
                str(item["elements"]),
                duration(item["host_total_seconds"]),
                duration(item["device_seconds"]),
                duration(item["host_overhead_seconds"]),
                duration(item.get("single_call_total_seconds")),
                (
                    "-"
                    if item.get("absorbed_fraction_of_barrier") is None
                    else f"{item['absorbed_fraction_of_barrier'] * 100:.0f}%"
                ),
                "yes" if item["hidden_synchronization"] else "no",
            )
            for item in derived["cuda_synchronization"][:limit]
        ],
        title=(
            "CUDA host latency versus device execution (absorbed = share "
            "of pre-queued device work the call waited for, so a high "
            "value means the call synchronizes)"
        ),
    )

    _table(
        ("backend", "layer", "subject", "elems", "f32", "f64", "f32/f64"),
        [
            (
                item["backend"],
                item["layer"],
                item["subject"],
                str(item["elements"]),
                duration(item["float32_seconds"]),
                duration(item["float64_seconds"]),
                ratio(item["float32_over_float64"]),
            )
            for item in sorted(
                derived["dtype_comparison"],
                key=lambda entry: -(entry["float32_over_float64"] or 0),
            )[:limit]
        ],
        title="float32 versus float64",
    )

    _table(
        ("case", "backend", "median", "spread", "min", "max"),
        [
            (
                item["name"],
                item["backend"],
                duration(item["median_seconds"]),
                f"{item['variability_percent']:.1f}%",
                duration(item["min_seconds"]),
                duration(item["max_seconds"]),
            )
            for item in derived["noisy_cases"][:limit]
        ],
        title="High-variance measurements",
    )

    unsupported = derived["classifications"]["unsupported"]
    if unsupported:
        print(f"\nUnsupported combinations: {len(unsupported)}")
    errors = derived["classifications"]["errors"]
    if errors:
        print(f"Errored or empty cases: {len(errors)}")
        for item in errors[:10]:
            first_line = (item.get("reason") or "").splitlines()
            print(
                f"  {item['backend']:>6} {item['name']}: "
                + (first_line[0] if first_line else "")
            )


def build_report(
    *,
    metadata: dict[str, Any],
    settings: dict[str, Any],
    records: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Assemble the full report, including every derived table."""
    return {
        "schema": "tensors-perfmap/1",
        "metadata": metadata,
        "settings": settings,
        "records": list(records),
        "analysis": analysis.build_analysis(records),
    }


__all__ = [
    "CSV_COLUMNS",
    "build_report",
    "byte_size",
    "duration",
    "print_summary",
    "ratio",
    "write_csv",
    "write_json",
    "write_samples_csv",
]

"""Flat exports: one row per record, and one row per raw sample."""

from __future__ import annotations

import csv
from collections.abc import Iterable
from pathlib import Path
from typing import Any

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
            "x".join(str(item) for item in shape) if isinstance(shape, list) else shape
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
        "absorbed_fraction_of_barrier": cuda.get("absorbed_fraction_of_barrier"),
        "hidden_synchronization": cuda.get("hidden_synchronization"),
        "work_items": record.get("work_items"),
        "work_items_per_second": record.get("work_items_per_second"),
        "host_peak_bytes": memory.get("host_peak_bytes"),
        "host_allocation_count": memory.get("host_allocation_count"),
        "host_retained_bytes_per_call": memory.get("host_retained_bytes_per_call"),
        "device_used_delta_bytes": memory.get("device_used_delta_bytes"),
        "device_retained_bytes_per_call": memory.get("device_retained_bytes_per_call"),
        "ladder": tags.get("ladder"),
        "curve": tags.get("curve"),
        "reason": (
            (record.get("reason") or "").splitlines()[0]
            if record.get("reason")
            else None
        ),
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
            [
                "name",
                "backend",
                "layer",
                "elements",
                "metric",
                "sample_index",
                "seconds",
                "loops_per_sample",
            ]
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
                    writer.writerow(
                        [
                            record["name"],
                            record["backend"],
                            record["layer"],
                            record.get("elements"),
                            metric,
                            index,
                            value,
                            loops,
                        ]
                    )


__all__ = [
    "CSV_COLUMNS",
    "write_csv",
    "write_samples_csv",
]

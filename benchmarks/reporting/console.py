"""Human-readable formatting and the summary printed after a run."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


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
        print("  ".join(cell.ljust(widths[index]) for index, cell in enumerate(row)))


def print_summary(report: dict[str, Any], *, limit: int = 20) -> None:
    """Print the tables that make a run readable without opening the JSON."""
    derived = report["analysis"]
    counts = derived["classifications"]["counts"]
    print(
        "\nRecords: "
        + ", ".join(f"{key}={value}" for key, value in sorted(counts.items()))
    )

    _table(
        (
            "ladder",
            "backend",
            "dtype",
            "elems",
            "base",
            "top",
            "ratio",
            "responsible step",
        ),
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
        (
            "case",
            "elems",
            "host",
            "device",
            "host-device",
            "1-call",
            "absorbed",
            "blocks",
        ),
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


__all__ = [
    "byte_size",
    "duration",
    "print_summary",
    "ratio",
]

"""Tables that put two measurements of the same work side by side.

A comparison needs the two sides to differ in exactly one thing, so
each of these finds its pairs by a tag the case declared rather than
by guessing from the name.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from .analysis import _ratio, measured, median_of


def _paired(
    records: Iterable[dict[str, Any]],
    tag: str,
    left_value: str,
    right_value: str,
) -> list[dict[str, Any]]:
    """Compare records that differ only in one tag's value."""
    grouped: dict[tuple[str, ...], dict[str, dict[str, Any]]] = {}
    for record in measured(records):
        tags = record.get("tags", {})
        value = tags.get(tag)
        if value not in (left_value, right_value):
            continue
        pair_key = tags.get("pair")
        if not pair_key:
            continue
        key = (record["backend"], record["layer"], pair_key)
        grouped.setdefault(key, {})[value] = record
    results = []
    for (backend, layer, pair_key), sides in sorted(grouped.items()):
        left = sides.get(left_value)
        right = sides.get(right_value)
        if left is None or right is None:
            continue
        results.append(
            {
                "backend": backend,
                "layer": layer,
                "pair": pair_key,
                left_value: {
                    "name": left["name"],
                    "median_seconds": median_of(left),
                    "noisy": left["host_total"]["noisy"],
                },
                right_value: {
                    "name": right["name"],
                    "median_seconds": median_of(right),
                    "noisy": right["host_total"]["noisy"],
                },
                "ratio": _ratio(median_of(right), median_of(left)),
                "difference_seconds": median_of(right) - median_of(left),
                "elements": left.get("elements"),
            }
        )
    return results


def crossovers(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Find, per curve, the size at which one backend overtakes another.

    A crossover is reported as the bracketing pair of measured sizes rather
    than an interpolated point, because only the measured sizes were
    observed.
    """
    grouped: dict[tuple[str, str, str], dict[str, dict[int, float]]] = {}
    for record in measured(records):
        curve = record.get("tags", {}).get("curve")
        if not curve or record.get("elements") is None:
            continue
        key = (record["layer"], curve, record.get("dtype") or "-")
        grouped.setdefault(key, {}).setdefault(record["backend"], {})[
            record["elements"]
        ] = median_of(record)

    pairs = (("python", "numpy"), ("python", "cuda"), ("numpy", "cuda"))
    results = []
    for (layer, curve, dtype), by_backend in sorted(grouped.items()):
        for slower, faster in pairs:
            left = by_backend.get(slower)
            right = by_backend.get(faster)
            if not left or not right:
                continue
            shared = sorted(set(left) & set(right))
            if len(shared) < 2:
                continue
            crossing = None
            for lower, upper in zip(shared, shared[1:]):
                before = left[lower] <= right[lower]
                after = left[upper] <= right[upper]
                if before and not after:
                    crossing = {
                        "between_elements": [lower, upper],
                        "ratio_before": _ratio(right[lower], left[lower]),
                        "ratio_after": _ratio(right[upper], left[upper]),
                    }
                    break
            results.append(
                {
                    "layer": layer,
                    "curve": curve,
                    "dtype": dtype,
                    "baseline": slower,
                    "candidate": faster,
                    "crossover": crossing,
                    "candidate_faster_at": [
                        size for size in shared if right[size] < left[size]
                    ],
                    "candidate_slower_at": [
                        size for size in shared if right[size] >= left[size]
                    ],
                    "speedup_at_largest": _ratio(left[shared[-1]], right[shared[-1]]),
                }
            )
    return results


def dtype_comparison(
    records: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Compare float32 against float64 for otherwise identical work."""
    grouped: dict[tuple[str, str, str, Any], dict[str, dict[str, Any]]] = {}
    for record in measured(records):
        dtype = record.get("dtype")
        if dtype not in ("float32", "float64"):
            continue
        key = (
            record["backend"],
            record["layer"],
            record.get("tags", {}).get("dtype_pair")
            or record.get("tags", {}).get("ladder", "").rsplit("|", 2)[0]
            or record["family"],
            record.get("elements"),
        )
        grouped.setdefault(key, {})[dtype] = record
    results = []
    for (backend, layer, subject, elements), sides in sorted(
        grouped.items(),
        key=lambda item: (
            str(item[0][0]),
            str(item[0][1]),
            str(item[0][2]),
            item[0][3] or 0,
        ),
    ):
        single = sides.get("float32")
        double = sides.get("float64")
        if single is None or double is None:
            continue
        results.append(
            {
                "backend": backend,
                "layer": layer,
                "subject": subject,
                "elements": elements,
                "float32_seconds": median_of(single),
                "float64_seconds": median_of(double),
                "float32_over_float64": _ratio(median_of(single), median_of(double)),
                "float32_case": single["name"],
                "float64_case": double["name"],
                "noisy": (
                    single["host_total"]["noisy"] or double["host_total"]["noisy"]
                ),
            }
        )
    return results


def layout_comparison(
    records: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Compare non-contiguous layouts against their contiguous baseline."""
    grouped: dict[tuple[str, str, Any], dict[str, dict[str, Any]]] = {}
    for record in measured(records):
        layout = record.get("tags", {}).get("layout")
        if not layout:
            continue
        key = (
            record["backend"],
            record.get("tags", {}).get("pair", record["family"]),
            record.get("elements"),
        )
        grouped.setdefault(key, {})[layout] = record
    results = []
    for (backend, subject, elements), layouts in sorted(
        grouped.items(),
        key=lambda item: (str(item[0][0]), str(item[0][1]), item[0][2] or 0),
    ):
        baseline = layouts.get("contiguous")
        if baseline is None:
            continue
        for layout, record in sorted(layouts.items()):
            if layout == "contiguous":
                continue
            results.append(
                {
                    "backend": backend,
                    "subject": subject,
                    "elements": elements,
                    "layout": layout,
                    "contiguous_seconds": median_of(baseline),
                    "layout_seconds": median_of(record),
                    "ratio": _ratio(median_of(record), median_of(baseline)),
                    "difference_seconds": median_of(record) - median_of(baseline),
                    "case": record["name"],
                    "noisy": record["host_total"]["noisy"],
                }
            )
    return results


def forward_backward(
    records: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Compare each differentiated path against its own forward pass."""
    return _paired(records, "phase", "forward", "backward")


def trace_replay(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compare graph tracing against replay of the compiled program."""
    return _paired(records, "phase", "trace", "replay")


def guard_comparison(
    records: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Compare the reduction guard's fast path against its full path."""
    return _paired(records, "data", "same-sign", "mixed-sign")


__all__ = [
    "crossovers",
    "dtype_comparison",
    "forward_backward",
    "guard_comparison",
    "layout_comparison",
    "trace_replay",
]

"""Tables derived from one run's records.

A record says how long something took. These say what that means:
which layer a cost enters at, how a cost scales, where a backend
overtakes another, and which measurements are too noisy to read.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

#: The execution stack, outermost cost last. A ladder's ratios are taken
#: between neighbouring layers present in that ladder.
LAYER_ORDER: tuple[str, ...] = (
    "provider",
    "kernel",
    "dispatch",
    "public",
    "variable",
    "graph-trace",
    "graph-compile",
    "graph-replay",
    "autograd",
    "optimizer",
    "training",
)

_LAYER_RANK = {layer: index for index, layer in enumerate(LAYER_ORDER)}


def measured(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return only the records that produced timings."""
    return [
        record
        for record in records
        if record.get("classification") == "measured" and record.get("host_total")
    ]


def median_of(record: dict[str, Any]) -> float:
    """Return the comparison statistic for one record."""
    return record["host_total"]["median_seconds"]


def _ratio(numerator: float, denominator: float) -> float | None:
    if denominator <= 0.0:
        return None
    return numerator / denominator


def ladders(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Group records sharing a ``ladder`` tag and resolve their layer ratios.

    A ladder answers the attribution question directly: the same computation
    measured at several depths, with the step between each depth named.
    """
    grouped: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}
    for record in measured(records):
        key = record.get("tags", {}).get("ladder")
        if not key:
            continue
        rungs = grouped.setdefault((record["backend"], key), {})
        layer = record["layer"]
        # A ladder holds one measurement per layer; keep the faster one if a
        # suite offered two spellings of the same rung.
        existing = rungs.get(layer)
        if existing is None or median_of(record) < median_of(existing):
            rungs[layer] = record
    results: list[dict[str, Any]] = []
    for (backend, key), rungs in sorted(grouped.items()):
        present = sorted(
            rungs.values(),
            key=lambda item: _LAYER_RANK.get(item["layer"], 99),
        )
        if len(present) < 2:
            continue
        entry: dict[str, Any] = {
            "ladder": key,
            "backend": backend,
            "dtype": present[0].get("dtype"),
            "elements": present[0].get("elements"),
            "layers": {
                item["layer"]: {
                    "name": item["name"],
                    "median_seconds": median_of(item),
                    "noisy": item["host_total"]["noisy"],
                    "device_seconds": (item.get("device", {}).get("median_seconds")),
                }
                for item in present
            },
            "steps": [],
        }
        for lower, upper in zip(present, present[1:]):
            lower_time = median_of(lower)
            upper_time = median_of(upper)
            entry["steps"].append(
                {
                    "from": lower["layer"],
                    "to": upper["layer"],
                    "ratio": _ratio(upper_time, lower_time),
                    "added_seconds": upper_time - lower_time,
                }
            )
        base = present[0]
        top = present[-1]
        entry["total"] = {
            "from": base["layer"],
            "to": top["layer"],
            "ratio": _ratio(median_of(top), median_of(base)),
            "added_seconds": median_of(top) - median_of(base),
        }
        results.append(entry)
    return results


def scaling_curves(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return per-(backend, layer, family, dtype, curve) size series.

    A curve is what separates fixed overhead from scaling behavior: the
    intercept at one element is the cost of asking, and the slope at the top
    is the cost of the work.
    """
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    for record in measured(records):
        curve = record.get("tags", {}).get("curve")
        if not curve or record.get("elements") is None:
            continue
        key = (
            record["backend"],
            record["layer"],
            curve,
            record.get("dtype") or "-",
        )
        grouped.setdefault(key, []).append(record)
    results = []
    for (backend, layer, curve, dtype), items in sorted(grouped.items()):
        points = sorted(items, key=lambda item: item["elements"])
        if len(points) < 2:
            continue
        series = [
            {
                "elements": point["elements"],
                "median_seconds": median_of(point),
                "seconds_per_element": (
                    median_of(point) / point["elements"] if point["elements"] else None
                ),
                "noisy": point["host_total"]["noisy"],
            }
            for point in points
        ]
        smallest = points[0]
        largest = points[-1]
        # Fixed overhead is what a size curve is for: the smallest point is
        # almost entirely overhead, and comparing per-element cost at both
        # ends says how much of it survives at scale.
        results.append(
            {
                "backend": backend,
                "layer": layer,
                "curve": curve,
                "dtype": dtype,
                "series": series,
                "fixed_overhead_seconds": median_of(smallest),
                "elements_range": [smallest["elements"], largest["elements"]],
                "growth_ratio": _ratio(median_of(largest), median_of(smallest)),
                "size_ratio": (
                    largest["elements"] / smallest["elements"]
                    if smallest["elements"]
                    else None
                ),
            }
        )
    return results


#: The phases a training step is decomposed into, in execution order.
TRAINING_PHASES: tuple[str, ...] = ("forward", "loss", "backward", "optimizer")


def training_breakdown(
    records: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Split each measured training step into its phases.

    The combined step is measured separately from the phases, so comparing
    their sum against it says whether the decomposition accounts for the
    step or whether something sits between the phases.
    """
    grouped: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}
    for record in measured(records):
        tags = record.get("tags", {})
        phase = tags.get("phase")
        pair = tags.get("pair")
        if not phase or not pair or not pair.startswith("training-"):
            continue
        grouped.setdefault((record["backend"], pair), {})[phase] = record

    results = []
    for (backend, pair), phases in sorted(grouped.items()):
        step = phases.get("step")
        if step is None:
            continue
        present = {
            name: median_of(phases[name]) for name in TRAINING_PHASES if name in phases
        }
        if not present:
            continue
        total = sum(present.values())
        step_time = median_of(step)
        sustained = phases.get("sustained")
        entry: dict[str, Any] = {
            "backend": backend,
            "subject": pair.removeprefix("training-"),
            "dtype": step.get("dtype"),
            "elements": step.get("elements"),
            "parameters": step.get("tags", {}).get("parameters"),
            "step_seconds": step_time,
            "phase_sum_seconds": total,
            "unaccounted_seconds": step_time - total,
            "accounted_fraction": (None if step_time <= 0 else total / step_time),
            "phases": {
                name: {
                    "seconds": value,
                    "share_of_step": (None if step_time <= 0 else value / step_time),
                }
                for name, value in present.items()
            },
            "noisy": step["host_total"]["noisy"],
        }
        if sustained is not None:
            # Ten steps in one call, divided out: a per-step figure that
            # includes whatever accumulates across iterations.
            entry["ten_step_mean_seconds"] = median_of(sustained) / 10.0
            entry["sustained_over_single"] = (
                None if step_time <= 0 else (median_of(sustained) / 10.0) / step_time
            )
        results.append(entry)
    results.sort(key=lambda item: -item["step_seconds"])
    return results


def cuda_synchronization(
    records: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Rank CUDA records by how much host time exceeded device execution."""
    results = []
    for record in measured(records):
        cuda = record.get("cuda")
        if not cuda:
            continue
        total = median_of(record)
        device = record["device"]["median_seconds"]
        results.append(
            {
                "name": record["name"],
                "layer": record["layer"],
                "family": record["family"],
                "dtype": record.get("dtype"),
                "elements": record.get("elements"),
                "host_total_seconds": total,
                "host_submit_seconds": record["host_submit"]["median_seconds"],
                "device_seconds": device,
                "host_overhead_seconds": total - device,
                "device_utilization": _ratio(device, total),
                "single_call_submit_seconds": cuda.get("single_call_submit_seconds"),
                "single_call_total_seconds": cuda.get("single_call_total_seconds"),
                "absorbed_fraction_of_barrier": cuda.get(
                    "absorbed_fraction_of_barrier"
                ),
                "hidden_synchronization": cuda["hidden_synchronization"],
                "noisy": record["host_total"]["noisy"],
            }
        )
    results.sort(key=lambda item: -item["host_overhead_seconds"])
    return results


def memory_findings(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return allocation records ordered by retained bytes per call."""
    results = []
    for record in measured(records):
        memory = record.get("memory")
        if not memory or "error" in memory:
            continue
        results.append(
            {
                "name": record["name"],
                "backend": record["backend"],
                "layer": record["layer"],
                "family": record["family"],
                "elements": record.get("elements"),
                "host_peak_bytes": memory["host_peak_bytes"],
                "host_allocation_count": memory["host_allocation_count"],
                "host_retained_bytes_per_call": memory["host_retained_bytes_per_call"],
                "device_used_delta_bytes": memory["device_used_delta_bytes"],
                "device_retained_bytes_per_call": memory[
                    "device_retained_bytes_per_call"
                ],
            }
        )
    results.sort(key=lambda item: -abs(item["host_retained_bytes_per_call"] or 0.0))
    return results


def noisy_cases(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return every measurement whose spread crosses the noise threshold."""
    results = []
    for record in measured(records):
        statistics_block = record["host_total"]
        if not statistics_block["noisy"]:
            continue
        results.append(
            {
                "name": record["name"],
                "backend": record["backend"],
                "layer": record["layer"],
                "elements": record.get("elements"),
                "median_seconds": statistics_block["median_seconds"],
                "variability_percent": statistics_block["variability_percent"],
                "min_seconds": statistics_block["min_seconds"],
                "max_seconds": statistics_block["max_seconds"],
                "spread_ratio": _ratio(
                    statistics_block["max_seconds"],
                    statistics_block["min_seconds"],
                ),
                "samples_seconds": statistics_block["samples_seconds"],
            }
        )
    results.sort(key=lambda item: -item["variability_percent"])
    return results


def classification_summary(
    records: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    """Count records by classification and list the unsupported reasons."""
    counts: dict[str, int] = {}
    unsupported: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for record in records:
        classification = record.get("classification", "unknown")
        counts[classification] = counts.get(classification, 0) + 1
        if classification == "unsupported":
            unsupported.append(
                {
                    "name": record["name"],
                    "backend": record["backend"],
                    "suite": record["suite"],
                    "reason": record.get("reason"),
                }
            )
        elif classification in ("error", "skipped"):
            errors.append(
                {
                    "name": record["name"],
                    "backend": record["backend"],
                    "suite": record["suite"],
                    "classification": classification,
                    "reason": record.get("reason"),
                }
            )
    return {
        "counts": counts,
        "unsupported": unsupported,
        "errors": errors,
    }


#: A ladder step is only worth ranking when the workload is large enough for
#: the absolute cost to matter. Ratios on a one-element operation are
#: reported but never ranked by ratio alone.
RANKABLE_MINIMUM_ELEMENTS = 1_000


def bottlenecks(
    records: Iterable[dict[str, Any]],
    ladder_entries: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Build the candidate bottleneck table and rank it several ways.

    Each candidate is one ladder: the same computation measured at adjacent
    layers, so the absolute and multiplicative overhead attributed to a
    layer both come from the same observation.
    """
    candidates: list[dict[str, Any]] = []
    for entry in ladder_entries:
        layers = entry["layers"]
        base_layer, base = min(
            layers.items(), key=lambda item: _LAYER_RANK.get(item[0], 99)
        )
        top_layer, top = max(
            layers.items(), key=lambda item: _LAYER_RANK.get(item[0], 99)
        )
        # The dominant step is the one to name: it says which layer owns the
        # cost, rather than only that the stack as a whole is slow.
        dominant = max(
            entry["steps"],
            key=lambda step: step["added_seconds"],
            default=None,
        )
        elements = entry.get("elements") or 0
        candidates.append(
            {
                "ladder": entry["ladder"],
                "backend": entry["backend"],
                "dtype": entry.get("dtype"),
                "elements": elements,
                "base_layer": base_layer,
                "base_seconds": base["median_seconds"],
                "top_layer": top_layer,
                "top_seconds": top["median_seconds"],
                "absolute_overhead_seconds": (
                    top["median_seconds"] - base["median_seconds"]
                ),
                "multiplicative_overhead": entry["total"]["ratio"],
                "responsible_layer": (
                    f"{dominant['from']} to {dominant['to']}" if dominant else None
                ),
                "responsible_added_seconds": (
                    dominant["added_seconds"] if dominant else None
                ),
                "layer_times": {
                    name: rung["median_seconds"] for name, rung in layers.items()
                },
                "noisy": any(rung["noisy"] for rung in layers.values()),
                "rankable": elements >= RANKABLE_MINIMUM_ELEMENTS,
            }
        )

    def ranked(
        key: Any,
        *,
        require_rankable: bool = False,
        limit: int = 25,
    ) -> list[dict[str, Any]]:
        pool = [item for item in candidates if not require_rankable or item["rankable"]]
        return sorted(pool, key=key)[:limit]

    fixed_overhead = [item for item in candidates if (item["elements"] or 0) <= 10]
    return {
        "candidates": candidates,
        "by_absolute_runtime": ranked(
            lambda item: -item["top_seconds"], require_rankable=True
        ),
        "by_overhead_ratio": ranked(
            lambda item: -(item["multiplicative_overhead"] or 0.0),
            require_rankable=True,
        ),
        "by_absolute_overhead": ranked(
            lambda item: -item["absolute_overhead_seconds"],
            require_rankable=True,
        ),
        "by_fixed_overhead": sorted(
            fixed_overhead,
            key=lambda item: -item["absolute_overhead_seconds"],
        )[:25],
    }


__all__ = [
    "LAYER_ORDER",
    "RANKABLE_MINIMUM_ELEMENTS",
    "TRAINING_PHASES",
    "bottlenecks",
    "classification_summary",
    "cuda_synchronization",
    "ladders",
    "measured",
    "median_of",
    "memory_findings",
    "noisy_cases",
    "scaling_curves",
    "training_breakdown",
]

"""Render a merged performance map as the report's markdown tables.

The report is generated from the recorded data rather than transcribed from
it, so every figure in it is traceable to a record.

Usage::

    python -m benchmarks.perfmap.tables benchmarks/results/perfmap.json \
        --section all > report-tables.md
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any

from .report import byte_size, duration, ratio


def _markdown(
    heading: Sequence[str],
    rows: Iterable[Sequence[str]],
    *,
    title: str = "",
    note: str = "",
) -> str:
    rows = list(rows)
    lines: list[str] = []
    if title:
        lines.append(f"### {title}\n")
    if note:
        lines.append(f"{note}\n")
    if not rows:
        lines.append("_No measurements matched._\n")
        return "\n".join(lines)
    lines.append("| " + " | ".join(heading) + " |")
    lines.append("|" + "|".join("---" for _ in heading) + "|")
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    lines.append("")
    return "\n".join(lines)


def _flag(value: Any) -> str:
    return "yes" if value else ""


def _percent(value: float | None) -> str:
    return "-" if value is None else f"{value * 100:.0f}%"


def _elements(value: Any) -> str:
    if value is None:
        return "-"
    return f"{int(value):,}"


# -- sections ------------------------------------------------------------


def ladder_table(report: dict[str, Any], *, limit: int = 60) -> str:
    """Provider to public to higher-layer ratios, worst absolute first."""
    entries = report["analysis"]["ladders"]
    rankable = [
        entry for entry in entries
        if (entry.get("elements") or 0) >= 1_000
    ]
    rankable.sort(
        key=lambda entry: -(entry["total"]["added_seconds"] or 0.0)
    )
    rows = []
    for entry in rankable[:limit]:
        layers = entry["layers"]
        rows.append((
            entry["ladder"],
            entry["backend"],
            entry.get("dtype") or "-",
            _elements(entry.get("elements")),
            duration(layers.get("provider", {}).get("median_seconds")),
            duration(layers.get("kernel", {}).get("median_seconds")),
            duration(layers.get("dispatch", {}).get("median_seconds")),
            duration(layers.get("public", {}).get("median_seconds")),
            duration(layers.get("variable", {}).get("median_seconds")),
            duration(layers.get("graph-replay", {}).get("median_seconds")),
            ratio(entry["total"]["ratio"]),
            duration(entry["total"]["added_seconds"]),
            _flag(entry.get("steps") and any(
                rung.get("noisy") for rung in layers.values()
            )),
        ))
    return _markdown(
        ("ladder", "backend", "dtype", "elements", "provider", "kernel",
         "dispatch", "public", "variable", "replay", "total ratio",
         "added", "noisy"),
        rows,
        title="Layer ladders, ranked by absolute added time",
        note=(
            "One row is one computation measured at every depth it could be "
            "isolated at. `total ratio` and `added` compare the outermost "
            "measured layer against the innermost. Rows below 1,000 "
            "elements are excluded from this ranking and appear in the "
            "fixed-overhead table instead."
        ),
    )


def fixed_overhead_table(report: dict[str, Any], *, limit: int = 40) -> str:
    """The per-call floor, from the smallest point of each ladder."""
    entries = [
        entry for entry in report["analysis"]["ladders"]
        if (entry.get("elements") or 0) <= 10
    ]
    entries.sort(
        key=lambda entry: -(entry["total"]["added_seconds"] or 0.0)
    )
    rows = []
    for entry in entries[:limit]:
        layers = entry["layers"]
        rows.append((
            entry["ladder"],
            entry["backend"],
            entry.get("dtype") or "-",
            duration(layers.get("provider", {}).get("median_seconds")),
            duration(layers.get("kernel", {}).get("median_seconds")),
            duration(layers.get("public", {}).get("median_seconds")),
            duration(layers.get("variable", {}).get("median_seconds")),
            ratio(entry["total"]["ratio"]),
            duration(entry["total"]["added_seconds"]),
        ))
    return _markdown(
        ("ladder", "backend", "dtype", "provider", "kernel", "public",
         "variable", "ratio", "added"),
        rows,
        title="Fixed per-call overhead (1 to 10 elements)",
        note=(
            "At these sizes the arithmetic is free, so the whole cost is "
            "the machinery. These are floors, not optimization targets on "
            "their own."
        ),
    )


def scaling_table(report: dict[str, Any], *, limit: int = 80) -> str:
    """Size curves, as fixed cost plus per-element cost at both ends."""
    curves = report["analysis"]["scaling_curves"]
    rows = []
    for curve in sorted(
        curves,
        key=lambda item: (item["curve"], item["backend"], item["layer"]),
    )[:limit]:
        series = curve["series"]
        smallest, largest = series[0], series[-1]
        rows.append((
            curve["curve"],
            curve["backend"],
            curve["layer"],
            curve["dtype"],
            _elements(smallest["elements"]),
            duration(smallest["median_seconds"]),
            _elements(largest["elements"]),
            duration(largest["median_seconds"]),
            ratio(curve["growth_ratio"]),
            f"{curve['size_ratio']:,.0f}x" if curve["size_ratio"] else "-",
        ))
    return _markdown(
        ("curve", "backend", "layer", "dtype", "min n", "time at min n",
         "max n", "time at max n", "time growth", "size growth"),
        rows,
        title="Scaling curves",
        note=(
            "Where `time growth` is far below `size growth`, the small end "
            "is dominated by fixed cost. Where they match, the operation is "
            "scaling with its work."
        ),
    )


def crossover_table(report: dict[str, Any], *, limit: int = 60) -> str:
    """Where one backend overtakes another."""
    rows = []
    for entry in report["analysis"]["crossovers"]:
        crossing = entry["crossover"]
        faster = entry["candidate_faster_at"]
        rows.append((
            entry["curve"],
            entry["layer"],
            entry["dtype"],
            f"{entry['baseline']} to {entry['candidate']}",
            (
                f"{crossing['between_elements'][0]:,} to "
                f"{crossing['between_elements'][1]:,}"
                if crossing
                else ("always" if faster and not entry["candidate_slower_at"]
                      else "never")
            ),
            _elements(min(faster)) if faster else "-",
            ratio(entry["speedup_at_largest"]),
        ))
    rows.sort(key=lambda row: (row[3], row[0]))
    return _markdown(
        ("curve", "layer", "dtype", "comparison", "crossover between",
         "first size where faster", "speedup at largest size"),
        rows[:limit],
        title="Backend crossover points",
        note=(
            "A crossover is reported as the bracketing pair of measured "
            "sizes, because only those sizes were observed. `never` means "
            "the candidate was slower at every shared size."
        ),
    )


def dtype_table(report: dict[str, Any], *, limit: int = 60) -> str:
    """float32 against float64 on identical work."""
    entries = report["analysis"]["dtype_comparison"]
    entries = [
        entry for entry in entries
        if (entry.get("elements") or 0) >= 1_000
    ]
    entries.sort(key=lambda entry: -(entry["float32_over_float64"] or 0))
    rows = [
        (
            entry["subject"],
            entry["backend"],
            entry["layer"],
            _elements(entry.get("elements")),
            duration(entry["float64_seconds"]),
            duration(entry["float32_seconds"]),
            ratio(entry["float32_over_float64"]),
            _flag(entry["noisy"]),
        )
        for entry in entries[:limit]
    ]
    return _markdown(
        ("subject", "backend", "layer", "elements", "float64", "float32",
         "float32 / float64", "noisy"),
        rows,
        title="float32 versus float64",
        note=(
            "A ratio above 1 means float32 is slower than float64 for the "
            "same operation and size. On hardware whose float32 throughput "
            "exceeds its float64 throughput, that inversion is a library "
            "effect, not a hardware one."
        ),
    )


def layout_table(report: dict[str, Any], *, limit: int = 60) -> str:
    """Non-contiguous layouts against their contiguous baseline."""
    entries = report["analysis"]["layout_comparison"]
    entries.sort(key=lambda entry: -(entry["ratio"] or 0))
    rows = [
        (
            entry["case"],
            entry["backend"],
            entry["layout"],
            _elements(entry.get("elements")),
            duration(entry["contiguous_seconds"]),
            duration(entry["layout_seconds"]),
            ratio(entry["ratio"]),
        )
        for entry in entries[:limit]
    ]
    return _markdown(
        ("case", "backend", "layout", "elements", "contiguous", "this layout",
         "ratio"),
        rows,
        title="Contiguous versus non-contiguous layouts",
        note=(
            "Non-contiguous tensors are reachable only through the internal "
            "metadata constructor: no public operation returns one, because "
            "every public shape operation materializes a compact copy."
        ),
    )


def phase_table(
    report: dict[str, Any],
    key: str,
    left: str,
    right: str,
    *,
    title: str,
    note: str = "",
    limit: int = 60,
) -> str:
    """A paired comparison table (forward/backward, trace/replay, ...)."""
    entries = report["analysis"][key]
    entries.sort(key=lambda entry: -(entry["ratio"] or 0))
    rows = [
        (
            entry["pair"],
            entry["backend"],
            _elements(entry.get("elements")),
            duration(entry[left]["median_seconds"]),
            duration(entry[right]["median_seconds"]),
            ratio(entry["ratio"]),
            duration(entry["difference_seconds"]),
            _flag(entry[left]["noisy"] or entry[right]["noisy"]),
        )
        for entry in entries[:limit]
    ]
    return _markdown(
        ("subject", "backend", "elements", left, right,
         f"{right} / {left}", "difference", "noisy"),
        rows,
        title=title,
        note=note,
    )


def training_table(report: dict[str, Any], *, limit: int = 60) -> str:
    """Training steps split into phases, with the residual named."""
    entries = report["analysis"]["training_breakdown"]
    rows = []
    for entry in entries[:limit]:
        phases = entry["phases"]

        def share(name: str) -> str:
            item = phases.get(name)
            if item is None:
                return "-"
            return (
                f"{duration(item['seconds'])} "
                f"({item['share_of_step'] * 100:.0f}%)"
                if item["share_of_step"] is not None
                else duration(item["seconds"])
            )

        rows.append((
            entry["subject"],
            entry["backend"],
            entry.get("dtype") or "-",
            f"{int(entry['parameters']):,}" if entry.get("parameters") else "-",
            duration(entry["step_seconds"]),
            share("forward"),
            share("loss"),
            share("backward"),
            share("optimizer"),
            duration(entry["unaccounted_seconds"]),
            (
                ratio(entry.get("sustained_over_single"))
                if entry.get("sustained_over_single") is not None else "-"
            ),
            _flag(entry["noisy"]),
        ))
    return _markdown(
        ("model / batch / hidden / depth", "backend", "dtype", "parameters",
         "full step", "forward", "loss", "backward", "optimizer",
         "unaccounted", "10-step / 1-step", "noisy"),
        rows,
        title="End-to-end training breakdown",
        note=(
            "Phases are measured separately from the combined step, so "
            "`unaccounted` is what the four phases do not explain. "
            "`10-step / 1-step` is ten consecutive steps divided by ten "
            "against one isolated step: above 1 means cost accumulates "
            "across iterations."
        ),
    )


def cuda_table(report: dict[str, Any], *, limit: int = 60) -> str:
    """Host latency against device execution, and who blocks."""
    entries = report["analysis"]["cuda_synchronization"]
    rows = [
        (
            entry["name"],
            entry["layer"],
            entry.get("dtype") or "-",
            _elements(entry.get("elements")),
            duration(entry["host_total_seconds"]),
            duration(entry["device_seconds"]),
            duration(entry["host_overhead_seconds"]),
            _percent(entry.get("device_utilization")),
            _percent(entry.get("absorbed_fraction_of_barrier")),
            _flag(entry["hidden_synchronization"]),
        )
        for entry in entries[:limit]
    ]
    return _markdown(
        ("case", "layer", "dtype", "elements", "host", "device",
         "host - device", "device share", "absorbed", "blocks"),
        rows,
        title="CUDA host latency versus device execution",
        note=(
            "`absorbed` is the share of pre-queued device work the call "
            "waited for; a high value means the call synchronizes. `blocks` "
            "is that verdict. Ranked by how much host time exceeded device "
            "execution."
        ),
    )


def blocking_table(report: dict[str, Any]) -> str:
    """Only the CUDA paths that block, grouped by family."""
    entries = [
        entry for entry in report["analysis"]["cuda_synchronization"]
        if entry["hidden_synchronization"]
    ]
    families: dict[str, list[dict[str, Any]]] = {}
    for entry in entries:
        families.setdefault(entry["family"], []).append(entry)
    rows = []
    for family, items in sorted(families.items()):
        worst = max(items, key=lambda item: item["host_overhead_seconds"])
        rows.append((
            family,
            str(len(items)),
            worst["name"],
            duration(worst["host_total_seconds"]),
            duration(worst["device_seconds"]),
            _percent(worst.get("absorbed_fraction_of_barrier")),
        ))
    return _markdown(
        ("family", "blocking cases", "worst case", "host", "device",
         "absorbed"),
        rows,
        title="CUDA paths that synchronize, by family",
        note=(
            "Each of these performs at least one host/device barrier. The "
            "device time shows how little of the latency is arithmetic."
        ),
    )


def memory_table(report: dict[str, Any], *, limit: int = 50) -> str:
    """Allocation and retention."""
    entries = report["analysis"]["memory"]
    rows = [
        (
            entry["name"],
            entry["backend"],
            _elements(entry.get("elements")),
            byte_size(entry["host_peak_bytes"]),
            f"{entry['host_allocation_count']:,}",
            byte_size(entry["host_retained_bytes_per_call"]),
            byte_size(entry["device_used_delta_bytes"]),
            byte_size(entry["device_retained_bytes_per_call"]),
        )
        for entry in entries[:limit]
    ]
    return _markdown(
        ("case", "backend", "elements", "host peak", "host allocations",
         "host retained / call", "device delta", "device retained / call"),
        rows,
        title="Memory: allocation and retention",
        note=(
            "Measured in a separate untimed pass. `retained / call` is what "
            "a batch of calls failed to release after collection, so a "
            "positive figure that persists is growth."
        ),
    )


def noise_table(report: dict[str, Any], *, limit: int = 50) -> str:
    """High-variance measurements, reported rather than hidden."""
    entries = report["analysis"]["noisy_cases"]
    rows = [
        (
            entry["name"],
            entry["backend"],
            _elements(entry.get("elements")),
            duration(entry["median_seconds"]),
            f"{entry['variability_percent']:.0f}%",
            duration(entry["min_seconds"]),
            duration(entry["max_seconds"]),
            ratio(entry["spread_ratio"]),
        )
        for entry in entries[:limit]
    ]
    return _markdown(
        ("case", "backend", "elements", "median", "MAD / median", "min",
         "max", "max / min"),
        rows,
        title="High-variance measurements",
        note=(
            "Every case whose median absolute deviation exceeded 15% of its "
            "median. Conclusions drawn from these rows should be treated as "
            "provisional."
        ),
    )


def classification_table(report: dict[str, Any]) -> str:
    """What was measured, what was not, and why."""
    summary = report["analysis"]["classifications"]
    counts = _markdown(
        ("classification", "records"),
        sorted(
            (key, f"{value:,}") for key, value in summary["counts"].items()
        ),
        title="Record classifications",
    )
    reasons: dict[str, list[str]] = {}
    for entry in summary["unsupported"]:
        reason = (entry.get("reason") or "").strip().split("\n")[0]
        reasons.setdefault(reason, []).append(
            f"{entry['backend']}:{entry['name']}"
        )
    grouped = _markdown(
        ("reason", "backends and cases", "count"),
        [
            (
                reason[:180],
                ", ".join(sorted({item.split(":", 1)[0] for item in cases})),
                str(len(cases)),
            )
            for reason, cases in sorted(
                reasons.items(), key=lambda item: -len(item[1])
            )
        ],
        title="Unsupported combinations, with the stated reason",
        note=(
            "Nothing was silently omitted: a combination the package cannot "
            "express is recorded with its reason."
        ),
    )
    errors = _markdown(
        ("case", "backend", "classification", "reason"),
        [
            (
                entry["name"],
                entry["backend"],
                entry["classification"],
                (entry.get("reason") or "").strip().split("\n")[0][:160],
            )
            for entry in summary["errors"]
        ],
        title="Cases that errored or produced no samples",
    )
    return "\n".join((counts, grouped, errors))


def bottleneck_table(report: dict[str, Any], *, limit: int = 25) -> str:
    """The ranked bottleneck table, ranked several ways."""
    bottlenecks = report["analysis"]["bottlenecks"]
    sections = []
    for key, title, note in (
        (
            "by_absolute_runtime",
            "Ranked by largest absolute runtime",
            "The slowest outermost measurement, at 1,000 elements or more.",
        ),
        (
            "by_absolute_overhead",
            "Ranked by largest absolute library overhead",
            "Outermost minus innermost measured layer, in seconds.",
        ),
        (
            "by_overhead_ratio",
            "Ranked by largest overhead ratio",
            "Outermost divided by innermost, at 1,000 elements or more so a "
            "ratio on a one-element operation cannot lead the table.",
        ),
        (
            "by_fixed_overhead",
            "Ranked by largest fixed overhead",
            "The same comparison at 10 elements or fewer, where the whole "
            "cost is the machinery.",
        ),
    ):
        rows = [
            (
                entry["ladder"],
                entry["backend"],
                entry.get("dtype") or "-",
                _elements(entry.get("elements")),
                duration(entry["base_seconds"]),
                duration(entry["top_seconds"]),
                duration(entry["absolute_overhead_seconds"]),
                ratio(entry["multiplicative_overhead"]),
                str(entry["responsible_layer"]),
                _flag(entry["noisy"]),
            )
            for entry in bottlenecks[key][:limit]
        ]
        sections.append(_markdown(
            ("path", "backend", "dtype", "elements",
             f"innermost", "outermost", "absolute overhead", "ratio",
             "responsible step", "noisy"),
            rows,
            title=title,
            note=note,
        ))
    return "\n".join(sections)


def inventory_table(report: dict[str, Any]) -> str:
    """What was measured: suites, layers, families, dtypes, and sizes."""
    records = report["records"]
    by_suite: dict[str, dict[str, Any]] = {}
    for record in records:
        entry = by_suite.setdefault(record["suite"], {
            "groups": set(), "cases": set(), "records": 0,
            "measured": 0, "backends": set(), "layers": set(),
            "families": set(), "dtypes": set(), "elements": set(),
        })
        entry["groups"].add(record["group"])
        entry["cases"].add(record["name"])
        entry["records"] += 1
        if record["classification"] == "measured":
            entry["measured"] += 1
            entry["backends"].add(record["backend"])
        entry["layers"].add(record["layer"])
        entry["families"].add(record["family"])
        if record.get("dtype"):
            entry["dtypes"].add(record["dtype"])
        if record.get("elements"):
            entry["elements"].add(int(record["elements"]))

    suites = _markdown(
        ("suite", "groups", "distinct cases", "records", "measured",
         "backends", "layers", "dtypes", "element range"),
        [
            (
                suite,
                f"{len(entry['groups']):,}",
                f"{len(entry['cases']):,}",
                f"{entry['records']:,}",
                f"{entry['measured']:,}",
                ", ".join(sorted(entry["backends"])) or "-",
                ", ".join(sorted(entry["layers"])),
                ", ".join(sorted(entry["dtypes"])) or "-",
                (
                    f"{min(entry['elements']):,} to "
                    f"{max(entry['elements']):,}"
                    if entry["elements"] else "-"
                ),
            )
            for suite, entry in sorted(by_suite.items())
        ],
        title="Suites measured",
    )

    by_layer: dict[str, int] = {}
    for record in records:
        if record["classification"] != "measured":
            continue
        by_layer[record["layer"]] = by_layer.get(record["layer"], 0) + 1
    layers = _markdown(
        ("layer", "measurements"),
        [
            (layer, f"{count:,}")
            for layer, count in sorted(
                by_layer.items(), key=lambda item: -item[1]
            )
        ],
        title="Measurements by layer",
    )

    by_family: dict[str, int] = {}
    for record in records:
        if record["classification"] != "measured":
            continue
        by_family[record["family"]] = by_family.get(record["family"], 0) + 1
    families = _markdown(
        ("operation family", "measurements"),
        [
            (family, f"{count:,}")
            for family, count in sorted(by_family.items())
        ],
        title="Measurements by operation family",
    )
    return "\n".join((suites, layers, families))


def backend_summary_table(report: dict[str, Any]) -> str:
    """Per-backend and per-layer medians, as an orientation table."""
    import statistics

    records = [
        record for record in report["records"]
        if record["classification"] == "measured" and record.get("host_total")
    ]
    grouped: dict[tuple[str, str], list[float]] = {}
    for record in records:
        key = (record["backend"], record["layer"])
        grouped.setdefault(key, []).append(
            record["host_total"]["median_seconds"]
        )
    rows = [
        (
            backend,
            layer,
            f"{len(values):,}",
            duration(statistics.median(values)),
            duration(min(values)),
            duration(max(values)),
        )
        for (backend, layer), values in sorted(grouped.items())
    ]
    return _markdown(
        ("backend", "layer", "measurements", "median of medians",
         "fastest", "slowest"),
        rows,
        title="Orientation: spread of measurements by backend and layer",
        note=(
            "This aggregates unlike workloads and is only for orientation — "
            "the attribution comes from the ladder tables."
        ),
    )


SECTIONS: dict[str, Callable[[dict[str, Any]], str]] = {
    "inventory": inventory_table,
    "backends": backend_summary_table,
    "ladders": ladder_table,
    "fixed": fixed_overhead_table,
    "scaling": scaling_table,
    "crossover": crossover_table,
    "dtype": dtype_table,
    "layout": layout_table,
    "forward-backward": lambda report: phase_table(
        report, "forward_versus_backward", "forward", "backward",
        title="Forward versus backward",
        note=(
            "Both sides replay the same already-traced graph, so the ratio "
            "is a property of the derivative rules rather than of tracing."
        ),
    ),
    "trace-replay": lambda report: phase_table(
        report, "trace_versus_replay", "trace", "replay",
        title="Graph trace versus compiled replay",
        note=(
            "A ratio below 1 means replay is cheaper than tracing the same "
            "computation, which is what compilation is for."
        ),
    ),
    "guard": lambda report: phase_table(
        report, "reduction_guard", "same-sign", "mixed-sign",
        title="Reduction stability guard: fast path versus full path",
        note=(
            "Same-sign finite data can take the guard's fast path; "
            "alternating-sign data cannot. The difference is the cost of "
            "the full guard."
        ),
    ),
    "training": training_table,
    "cuda": cuda_table,
    "blocking": blocking_table,
    "memory": memory_table,
    "noise": noise_table,
    "classification": classification_table,
    "bottlenecks": bottleneck_table,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m benchmarks.perfmap.tables",
        description="Render a performance map as markdown tables.",
    )
    parser.add_argument("input", type=Path)
    parser.add_argument(
        "--section",
        action="append",
        choices=("all", *SECTIONS),
        help="section to render; repeatable, defaults to all",
    )
    parser.add_argument("--limit", type=int, default=None)
    arguments = parser.parse_args(argv)

    report = json.loads(arguments.input.read_text(encoding="utf-8"))
    requested = arguments.section or ["all"]
    names = list(SECTIONS) if "all" in requested else requested
    for name in names:
        renderer = SECTIONS[name]
        if arguments.limit is not None:
            try:
                print(renderer(report, limit=arguments.limit))  # type: ignore[call-arg]
                continue
            except TypeError:
                pass
        print(renderer(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

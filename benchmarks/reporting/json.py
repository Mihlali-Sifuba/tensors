"""Assembling a report and writing it as JSON.

``build_report`` is where a run's records become a stored artifact:
the raw observations, the settings that produced them, the
environment, and every table derived from them. Keeping the raw
records beside the derived tables is what makes a stored run
re-analysable when the analysis changes.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from . import analysis, comparison


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


def build_analysis(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Return every derived table for one run's records.

    This is where the two halves of the analysis meet: the tables that
    describe one measurement and the tables that put two side by side. The
    keys are the stored report's, so they are part of its schema.
    """
    ladder_entries = analysis.ladders(records)
    return {
        "ladders": ladder_entries,
        "scaling_curves": analysis.scaling_curves(records),
        "crossovers": comparison.crossovers(records),
        "dtype_comparison": comparison.dtype_comparison(records),
        "layout_comparison": comparison.layout_comparison(records),
        "forward_versus_backward": comparison.forward_backward(records),
        "trace_versus_replay": comparison.trace_replay(records),
        "reduction_guard": comparison.guard_comparison(records),
        "training_breakdown": analysis.training_breakdown(records),
        "cuda_synchronization": analysis.cuda_synchronization(records),
        "memory": analysis.memory_findings(records),
        "noisy_cases": analysis.noisy_cases(records),
        "classifications": analysis.classification_summary(records),
        "bottlenecks": analysis.bottlenecks(records, ladder_entries),
    }


#: The version a report written today carries.
#:
#: ``/1`` labelled a record with the suite that produced it under the old
#: flat layout, where ``layers`` covered arithmetic, the elementary
#: functions, the activations, and the comparisons together, and ``memory``
#: was a suite rather than a measurement pass. ``/2`` labels it with the
#: semantic domain instead. The records themselves — their timings, samples,
#: layers, and families — did not change, so a ``/1`` report is still
#: readable; only what its ``suite`` and ``group`` strings refer to is.
SCHEMA = "tensors-perfmap/2"

#: Versions this package can read. A reader that accepts both must not
#: silently treat their suite labels as the same vocabulary.
READABLE_SCHEMAS: tuple[str, ...] = ("tensors-perfmap/1", SCHEMA)


def schema_of(report: dict[str, Any]) -> str:
    """Return a report's schema version, defaulting to the earliest."""
    return str(report.get("schema", READABLE_SCHEMAS[0]))


def build_report(
    *,
    metadata: dict[str, Any],
    settings: dict[str, Any],
    records: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Assemble the full report, including every derived table."""
    return {
        "schema": SCHEMA,
        "metadata": metadata,
        "settings": settings,
        "records": list(records),
        "analysis": build_analysis(records),
    }


__all__ = [
    "READABLE_SCHEMAS",
    "SCHEMA",
    "build_analysis",
    "build_report",
    "schema_of",
    "write_json",
]

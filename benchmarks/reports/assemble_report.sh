#!/usr/bin/env bash
# Assemble REPORT.md from the narrative files and the generated tables.
#
# Every table comes from benchmarks/reports/perfmap.json via
# `python -m benchmarks.reporting.tables`, so no figure in the report is
# transcribed by hand.
set -eu
cd "$(dirname "$0")/../.."
PY=".venv/Scripts/python.exe"
MAP="benchmarks/reports/perfmap.json"
OUT="benchmarks/reports/REPORT.md"
HERE="benchmarks/reports"

table() { "$PY" -m benchmarks.reporting.tables "$MAP" --section "$1"; }

{
  cat "$HERE/REPORT-header.md"

  echo
  echo "### Inventory"
  echo
  table inventory
  table backends

  echo "---"
  echo
  echo "## 4. Complete machine-readable results"
  echo
  cat "$HERE/RESULTS-INDEX.md"

  echo "---"
  echo
  echo "## 5. Summary tables by backend, layer, family, dtype and size"
  echo
  table scaling

  echo "---"
  echo
  echo "## 6. Provider to kernel to public overhead ratios"
  echo
  table ladders
  table fixed

  echo "---"
  echo
  echo "## 7. Scaling curves"
  echo
  echo "See the scaling table in section 5; the per-point series for every"
  echo 'curve is in `perfmap.json` under `analysis.scaling_curves`.'
  echo

  echo "---"
  echo
  echo "## 8. CPU / NumPy / CUDA crossover points"
  echo
  table crossover

  echo "---"
  echo
  echo "## 9. float32 versus float64"
  echo
  table dtype

  echo "---"
  echo
  echo "## 10. Contiguous versus non-contiguous"
  echo
  table layout

  echo "---"
  echo
  echo "## 11. Cold start versus warm"
  echo
  cat "$HERE/SECTION-coldstart.md"

  echo "---"
  echo
  echo "## 12. Graph trace versus replay"
  echo
  table trace-replay

  echo "---"
  echo
  echo "## 13. Forward versus backward"
  echo
  table forward-backward

  echo "---"
  echo
  echo "## 14. Optimizer scaling"
  echo
  cat "$HERE/SECTION-optimizer.md"

  echo "---"
  echo
  echo "## 15. End-to-end training breakdowns"
  echo
  table training

  echo "---"
  echo
  echo "## 16. Memory and allocation findings"
  echo
  table memory

  echo "---"
  echo
  echo "## 17. CUDA host latency versus device execution"
  echo
  table cuda
  table blocking

  echo "---"
  echo
  echo "## 18. High-variance and methodologically questionable cases"
  echo
  table noise
  table classification

  echo "---"
  echo
  echo "## 19. What cannot be meaningfully benchmarked, and why"
  echo
  cat "$HERE/COVERAGE.md"

  echo "---"
  echo
  cat "$HERE/SECTION-bottlenecks.md"

  echo "---"
  echo
  echo "## Appendix A: reduction stability guard, fast path versus full path"
  echo
  table guard

  echo "---"
  echo
  echo "## Appendix B: automatically generated bottleneck rankings"
  echo
  table bottlenecks

  echo "---"
  echo
  echo "## Appendix C: structural findings from reading the implementation"
  echo
  cat "$HERE/FINDINGS.md"

  echo "---"
  echo
  echo "## Appendix D: methodology in full"
  echo
  cat "$HERE/METHODOLOGY.md"

  echo "---"
  echo
  echo "## Appendix E: how the bottleneck ranking was built"
  echo
  cat "$HERE/BOTTLENECK-METHOD.md"

  echo "---"
  echo
  echo "## Appendix F: suites and cases"
  echo
  cat "$HERE/SUITES.md"
} > "$OUT"

echo "Wrote $OUT ($(wc -l < "$OUT") lines)"

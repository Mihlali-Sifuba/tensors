#!/usr/bin/env bash
# Finish the baseline: re-measure the suites whose readings are in doubt,
# merge everything, then run the diagnostic profiling.
#
# Two reasons a suite is re-measured:
#   dispatch, synchronization - measured first, while the machine was
#                    still settling
#   manipulation, storage - a short verification script shared the machine
#                    with
#                     them (a merge test and a profiling-path check)
#   creation        - stopped mid-suite once an ungated integer size was
#                     found to build ten million Python ints per call
#
# The first readings are kept under baseline/early/ as evidence; only the
# steady-state readings enter the merge.
set -u
OUT="benchmarks/reports/baseline"
LOG="$OUT/run.log"
PY=".venv/Scripts/python.exe"
ROUNDS="${ROUNDS:-5}"
TARGET="${TARGET:-0.02}"

if command -v tasklist >/dev/null 2>&1; then
  existing=$(tasklist /FI "IMAGENAME eq python.exe" /FO CSV 2>/dev/null | grep -c '^"python.exe"' || true)
  if [ "${existing:-0}" -gt 0 ]; then
    echo "refusing to start: $existing python process(es) already running" >&2
    exit 2
  fi
fi

mkdir -p "$OUT/early"
SUSPECT="dispatch synchronization manipulation storage creation"

for suite in $SUSPECT; do
  for extension in json csv; do
    [ -f "$OUT/$suite.$extension" ] && mv "$OUT/$suite.$extension" "$OUT/early/"
  done
  [ -f "$OUT/$suite-samples.csv" ] && mv "$OUT/$suite-samples.csv" "$OUT/early/"
done

echo "=== steady-state re-measurement $(date -u +%FT%TZ) ===" >> "$LOG"
for suite in $SUSPECT; do
  echo "########## $suite (steady state) ##########" >> "$LOG"
  start=$SECONDS
  "$PY" -m benchmarks --suite "$suite" \
    --rounds "$ROUNDS" --target-time "$TARGET" --memory \
    --output "$OUT/$suite.json" >> "$LOG" 2>&1
  echo "---------- $suite finished in $((SECONDS-start))s (exit $?) ----------" >> "$LOG"
done

echo "########## merge ##########" >> "$LOG"
"$PY" -m benchmarks.reporting.merge "$OUT/*.json" \
  --output benchmarks/reports/perfmap.json >> "$LOG" 2>&1
echo "---------- merge exit $? ----------" >> "$LOG"

echo "########## profiling ##########" >> "$LOG"
"$PY" -m benchmarks.reporting.profile_report \
  --output benchmarks/reports/profiling.json >> "$LOG" 2>&1
echo "---------- profiling exit $? ----------" >> "$LOG"
echo "=== finalize complete $(date -u +%FT%TZ) ===" >> "$LOG"

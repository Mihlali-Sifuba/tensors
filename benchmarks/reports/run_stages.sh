#!/usr/bin/env bash
# Run the performance map one suite at a time.
#
# Output goes to a log file rather than through a pipeline: a pipeline whose
# reader exits leaves the measuring process writing into a closed pipe, and
# on Windows the process tree is not reliably reaped when the shell is
# stopped. One process at a time, one log, no readers.
set -u
OUT="benchmarks/reports/baseline"
LOG="$OUT/run.log"
PY=".venv/Scripts/python.exe"
ROUNDS="${ROUNDS:-5}"
TARGET="${TARGET:-0.02}"
mkdir -p "$OUT"

# Refuse to start alongside another measuring process: two of them contend
# for the same cores and device and would corrupt both runs.
if command -v tasklist >/dev/null 2>&1; then
  existing=$(tasklist /FI "IMAGENAME eq python.exe" /FO CSV 2>/dev/null | grep -c '^"python.exe"' || true)
  if [ "${existing:-0}" -gt 0 ]; then
    echo "refusing to start: $existing python process(es) already running" >&2
    exit 2
  fi
fi

echo "=== run started $(date -u +%FT%TZ) rounds=$ROUNDS target=$TARGET ===" >> "$LOG"
for suite in "$@"; do
  echo "########## $suite ##########" >> "$LOG"
  start=$SECONDS
  "$PY" -m benchmarks --suite "$suite" \
    --rounds "$ROUNDS" --target-time "$TARGET" --memory \
    --output "$OUT/$suite.json" >> "$LOG" 2>&1
  status=$?
  echo "---------- $suite finished in $((SECONDS-start))s (exit $status) ----------" >> "$LOG"
done
echo "=== run complete $(date -u +%FT%TZ) ===" >> "$LOG"

#!/usr/bin/env bash
# Daily tournament cycle: refresh data + odds, regenerate predictions, score & tune on results.
# Run from anywhere; paths are resolved relative to this script.
set -euo pipefail
cd "$(dirname "$0")"

PY=".venv/bin/python"
STAMP="$(date +%Y%m%d_%H%M%S)"
mkdir -p logs
LOG="logs/run_${STAMP}.log"

{
  echo "=== run $STAMP ==="
  # 1. Refresh dataset (new results + next round) and regenerate blended predictions + live odds.
  "$PY" predict.py --refresh
  echo
  # 2. Score every saved prediction file against actual results, and tune the blend weight.
  "$PY" score.py --tune predictions_*.csv
} 2>&1 | tee "$LOG"

echo "Logged to $LOG"

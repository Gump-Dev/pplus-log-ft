#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/travix3/pplus-log-ft}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
DATA_DIR="${DATA_DIR:-./datasets}"
REPORT_DIR="${REPORT_DIR:-./eval/reports}"
EVAL_SET="${EVAL_SET:-./eval/fixed_eval_set.jsonl}"
PER_CATEGORY="${PER_CATEGORY:-200}"

cd "$PROJECT_DIR"
mkdir -p "$REPORT_DIR"

echo "=== Dataset quality gate ==="
"$PYTHON_BIN" eval/dataset_quality.py \
  --data-dir "$DATA_DIR" \
  --output-json "$REPORT_DIR/dataset_quality.json" \
  --output-md "$REPORT_DIR/dataset_quality.md"

echo
echo "=== Fixed eval set ==="
"$PYTHON_BIN" eval/build_eval_set.py \
  --input "$DATA_DIR/test.jsonl" \
  --output "$EVAL_SET" \
  --per-category "$PER_CATEGORY"

echo
echo "Quality report: $REPORT_DIR/dataset_quality.md"
echo "Eval set: $EVAL_SET"
